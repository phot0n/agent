import time
import psutil
from filelock import FileLock, Timeout

import redis
import docker
from agent.server import Server
from agent.bench import Bench
from agent.bench_stopper import HelperMixin


class Config:
    redis_queue_key: str = "bench_start_queue"
    redis_failed_hash_key: str = "bench_start_failed"
    redis_failed_hash_expiry_mins: int = 10
    check_interval_seconds: int = 60
    memory_reserve_percent: float = 35.0  # Reserve system memory
    memory_stats_file: str = "/home/frappe/agent/bench-memory-stats.json"
    worker_memory_mb: int = 100  # this is an estimate
    batch_size: int = 5
    activity_threshold_hours: float = 0.5  # Consider container active if accessed within last x hrs (for mem adjustment)
    min_uptime_hours: float = 0.25 # Minimum uptime before checking activity logs (for mem adjustment)
    available_memory_adjustment_percent: float = 30.0  # Max percent of available memory to adjust based on running containers


class BenchStarter(HelperMixin):
    def __init__(self):
        super().__init__(self)
        self.redis_client = None

    def queue_request(self, bench_name, ignore_throttle=False):
        if not self.redis_client:
            self._init_redis_client()

        # NOTE: Small race condition exists between checks and add.
        # Using nx=True prevents duplicates in queue/set, but item might
        # be added after being removed. It should be okay for the most part.

        status = "REQUEST_ALREADY_EXISTS"
        if not self.redis_client.zscore(Config.redis_queue_key, bench_name):
            status = "THROTTLED"
            if ignore_throttle or not self.redis_client.hget(f"{Config.redis_failed_hash_key}:{bench_name}", "throttle"):
                added = self.redis_client.zadd(Config.redis_queue_key, {bench_name: time.time()}, nx=True)
                status = "QUEUED" if added else "REQUEST_ALREADY_EXISTS"

        return status

    def _init_redis_client(self):
        try:
            self.redis_client = redis.Redis(
                port=self.server.config["redis_port"],
                decode_responses=True
            )
        except Exception as e:
            self._log(f"Failed to connect to Redis: {e}")
            raise

    def _get_system_memory_info(self):
        """Get system memory information including swap."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            'total_effective_bytes': memory.total + swap.total,
            'available_effective_bytes': memory.available + (swap.total - swap.used),
        }

    def _get_memory_threshold(self):
        # Use effective memory (RAM + swap) for threshold calculation
        total_effective_memory = self._get_system_memory_info()['total_effective_bytes']
        return int(total_effective_memory * (Config.memory_reserve_percent / 100))

    def _load_bench_config(self, bench_name):
        """Load bench configuration from config.json."""
        try:
            return Bench(bench_name, self.server).bench_config
        except Exception as e:
            self._log(f"Could not load config for bench {bench_name}: {e}")
        return {}

    def _calculate_memory_adjustment(self, bench):
        """Calculate memory adjustment based on activity."""
        try:
            # Check if container is active
            if not self._is_container_active(bench, Config.min_uptime_hours, Config.activity_threshold_hours):
                self._log(f"{bench.name} inactive (no recent web activity), skipping adjustment")
                return 0

            # Get current memory usage of the container if it's running
            current_usage = self._get_current_container_memory(bench)
            if not current_usage or current_usage <= 0:
                # Container not running or can't get stats
                return 0

            # underestimation (how much more memory container uses than current)
            return max(0, self.mem_stats[bench.name] - current_usage)

        except Exception as e:
            self._log(f"Error calculating memory adjustment for {bench.name}: {e}")

        return 0

    def _get_adjusted_available_memory(self):
        """Adjust available memory based on historical data of running containers."""

        total_adjustment = 0
        for bench in self.docker_client.containers.list():
            if bench.name not in self.mem_stats:
                continue  # No historical data to work with

            total_adjustment += self._calculate_memory_adjustment(bench)

        available_memory = self._get_system_memory_info()["available_effective_bytes"]
        max_adjustment = int(available_memory * (Config.available_memory_adjustment_percent / 100))
        if total_adjustment > max_adjustment:
            total_adjustment = max_adjustment

        return available_memory - total_adjustment

    def _calculate_bench_memory_requirement(self, bench_name):
        # First try to get from memory stats file
        if bench_name in self.mem_stats:
            mem_stat = self.mem_stats[bench_name]
            if mem_stat > 0:
                self._log(f"Using historical memory for {bench_name}: {mem_stat/(1024*1024)}MB")
                return mem_stat

        # use bench config to figure out the memory - this is an estimate at best
        bench_config = self._load_bench_config(bench_name)

        # Calculate based on workers
        background_workers = bench_config.get('background_workers', 1)
        gunicorn_workers = bench_config.get('gunicorn_workers', 2)
        merge_all_rq = bench_config.get('merge_all_rq_queues', False)
        merge_default_short = bench_config.get('merge_default_and_short_rq_queues', False)

        # Calculate total background workers based on queue merging
        if merge_all_rq:
            total_bg_workers = background_workers
        elif merge_default_short:
            total_bg_workers = 2 * background_workers
        else:
            total_bg_workers = 3 * background_workers

        total_workers = gunicorn_workers + total_bg_workers
        estimated_memory = total_workers * Config.worker_memory_mb
        self._log(f"Estimated memory for {bench_name} based on config: {estimated_memory}MB")
        return estimated_memory * 1024 * 1024

    def _can_start_bench(self, available_memory, min_available_threshold, required_memory_by_bench):
        # Check if current available memory is already below threshold
        if available_memory < min_available_threshold:
            return False, "Available memory below minimum threshold"

        # Check if starting this bench would drop available memory below threshold
        projected_available = available_memory - required_memory_by_bench
        if projected_available < min_available_threshold:
            return False, "Starting bench would drop available memory below threshold"

        return True, ""

    def _start_container(self, bench_name):
        try:
            with FileLock(f"/tmp/{bench_name}.lock", timeout=60):
                container = self.docker_client.containers.get(bench_name)
                if container.status in ("exited", "stopped"):
                    # start container and then start bench services
                    container.start()
                    Bench(container.name, self.server).docker_execute("supervisorctl start all", as_root=True)
                    self._log(f"Started container {bench_name}")
                    return "STARTED"
                elif container.status in ("running", "restarting"):
                    self._log(f"Container {bench_name} already running/restarting")
                    return "ALREADY_RUNNING"

            self._log(f"Unable to start {bench_name}, unknown container status: {container.status}")
        except docker.errors.NotFound:
            self._log(f"Container {bench_name} not found")
        except Timeout:
            self._log(f"Could not acquire lock for {bench_name}, skipping starting it")
        except Exception as e:
            self._log(f"Failed to start container {bench_name}: {e}")

        return "NOT_STARTED"

    def _get_pending_benches(self):
        """Get list of benches waiting to be started from Redis list."""
        try:
            # Get items from sorted set (based on ascending score)
            pending_items = self.redis_client.zrange(
                Config.redis_queue_key,
                0,
                Config.batch_size - 1
            )

            return [item.strip() for item in pending_items]
        except Exception as e:
            self._log(f"Error getting pending benches from Redis: {e}")

        return []

    def _remove_from_queue(self, bench_name, pipe=None):
        """Remove bench from the set."""
        try:
            client = pipe or self.redis_client
            client.zrem(Config.redis_queue_key, bench_name)
        except Exception as e:
            self._log(f"Error removing {bench_name} from start queue: {e}")

    def _remove_and_add_failed_status(self, bench_name, info, throttle=False):
        """Atomically remove bench from start queue and add failed status."""
        try:
            with self.redis_client.pipeline() as pipe:
                pipe.multi()

                self._remove_from_queue(bench_name, pipe)

                failed_key = f"{Config.redis_failed_hash_key}:{bench_name}"
                pipe.hset(failed_key, mapping={'info': info, 'throttle': int(throttle)})
                pipe.expire(failed_key, Config.redis_failed_hash_expiry_mins * 60) # set expiry

                pipe.execute()
        except Exception as e:
            self._log(f"Error moving {bench_name} to failed queue: {e}")

    def _process_batch(self):
        # Get pending benches from main queue
        pending_benches = self._get_pending_benches()
        if pending_benches:
            self.mem_stats = self._load_memory_stats(Config.memory_stats_file)

            min_available_threshold = self._get_memory_threshold()
            available_memory = self._get_system_memory_info()["available_effective_bytes"]

            # adjust if available mem is above threshold
            if available_memory > min_available_threshold:
                available_memory = self._get_adjusted_available_memory()

            self._log(f"""
                Available Memory (after adjustment if any): {available_memory/(1024*1024)}MB,
                Minimum Threshold: {min_available_threshold/(1024*1024)}MB
            """)

        for bench_name in pending_benches:
            if not self.running:
                break

            self._log(f"Processing {bench_name}")
            required_memory_by_bench = self._calculate_bench_memory_requirement(bench_name)

            throttle = True
            can_start, info = self._can_start_bench(available_memory, min_available_threshold, required_memory_by_bench)
            if can_start:
                status = self._start_container(bench_name)
                if status == "NOT_STARTED":
                    # dont throttle for non-memory related stuff
                    throttle = False
                    info = "Please try to queue again and/or contact support."
                    self._remove_and_add_failed_status(bench_name, info, throttle)
                else:
                    if status == "STARTED":
                        # reduce the available memory
                        available_memory = available_memory - required_memory_by_bench
                    self._remove_from_queue(bench_name)
            else:
                self._remove_and_add_failed_status(bench_name, info, throttle)

    def start(self):
        self._setup_signal_handlers()

        retries = 3
        self.running = True
        while self.running:
            try:
                self.sleeping = True
                time.sleep(Config.check_interval_seconds)
                self.sleeping = False

                self._log("Starting Bench Container Starter")

                self.server = Server()
                self._init_redis_client()
                self._init_docker_client()
                self._process_batch()

                self.mem_stats = None
                retries = 3
            except Exception as e:
                self._log(f"Unexpected error: {e}")

                retries -= 1
                if retries < 0:
                    self._log("Unable to recover - Exiting")
                    break

        self._log("Bench Starter stopped")


if __name__ == "__main__":
    BenchStarter().start()
