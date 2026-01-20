import sys
import os
import time
import datetime
import signal
import json
from pathlib import Path

import docker
import requests
from filelock import FileLock, Timeout

from agent.bench import Bench
from agent.server import Server


class HelperMixin:
    def __init__(self):
        self.running = False
        self.sleeping = False
        self.docker_client = None
        self.mem_stats = None
        self.server = None

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self._log(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        if self.sleeping:
            sys.exit(0)

    def _init_docker_client(self):
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            self._log(f"Failed to connect to Docker: {e}")
            raise

    def _load_memory_stats(self, stats_file):
        """Load memory stats from the stats file."""
        try:
            with FileLock("/tmp/bench_mem_stats.lock"):
                if os.path.exists(stats_file):
                    with open(stats_file, 'r') as f:
                        return json.load(f)
        except Exception as e:
            self._log(f"Could not load memory stats file: {e}")
        return {}

    def _log(self, message):
        print(f"[{datetime.datetime.now()}] {message!s}", flush=True)

    def _get_current_container_memory(self, container):
        try:
            stats = container.stats(stream=False)
            memory_stats = stats.get('memory_stats', {})
            return memory_stats.get('usage')
        except Exception as e:
            print(f"Could not get memory stats for {container.name}: {e}", flush=True)

    def _find_log_files_for_bench(self, bench_name):
        """Find log files associated with a bench."""
        log_files = []
        gunicorn_access_log_dir = Path(f"/home/frappe/benches/{bench_name}/logs")

        if not gunicorn_access_log_dir.exists():
            return

        for pattern in [
            "*-gunicorn.access.log",
        ]:
            found_files = list(gunicorn_access_log_dir.glob(pattern))
            log_files.extend(found_files)

        return set(log_files)

    def _get_last_activity_time(self, log_files):
        """Get the most recent modification time from log files."""
        latest_mtime = None

        for log_file in log_files:
            try:
                mtime = log_file.stat().st_mtime
                file_datetime = datetime.datetime.fromtimestamp(mtime)

                if latest_mtime is None or file_datetime > latest_mtime:
                    latest_mtime = file_datetime

            except Exception as e:
                self._log(f"Error checking modification time for {log_file}: {e}")

        return latest_mtime

    def _is_container_active(self, container, min_uptime_hours, inactive_threshold_hours):
        """Determine if a container should be stopped based on activity and uptime."""
        try:
            # Check container uptime
            start_time_str = container.attrs["State"]["StartedAt"]
            start_time = datetime.datetime.fromisoformat(
                start_time_str.rstrip('Z')[:26]
            ).replace(tzinfo=datetime.timezone.utc)

            now = datetime.datetime.now()
            uptime = now.astimezone(datetime.timezone.utc) - start_time

            # assume containers are active that haven't been running long enough
            if uptime < datetime.timedelta(hours=min_uptime_hours):
                self._log(f"Container {container.name} uptime {uptime} is below minimum threshold")
                return True

            # Find associated log files
            log_files = self._find_log_files_for_bench(container.name)
            if not log_files:
                self._log(f"No log files found for bench {container.name}")
                # just keep things active upto the threshold time and then convey them as inactive
                return uptime < datetime.timedelta(hours=inactive_threshold_hours)

            # Check last activity based on file modification times
            last_activity = self._get_last_activity_time(log_files)
            if not last_activity:
                self._log(f"Could not determine last activity for {container.name}")
                return uptime < datetime.timedelta(hours=inactive_threshold_hours)

            # Calculate time since last activity
            time_since_activity = now - last_activity
            return time_since_activity < datetime.timedelta(hours=inactive_threshold_hours)

        except Exception as e:
            self._log(f"Error in _is_container_active for container {container.name}: {e}")

        return False # assume that container is not active if there are any issues


class Config:
    check_interval_minutes: int = 15
    # this is the min uptime the container should have to start looking at the activity logs (for stopping the container(s) - assume them to be active if uptime less than this time)
    min_uptime_hours: float = 1.0
    # this is the max inactivity time for which if the container has not been accessed, it can be stopped
    inactive_threshold_hours: float = 2.0
    stats_file: str = "/home/frappe/agent/bench-memory-stats.json"
    cadvisor_endpoint: str = "http://127.0.0.1:9338/api/v1.3/docker"


class BenchStopper(HelperMixin):
    def __init__(self):
        super().__init__(self)
        self._setup_signal_handlers()

    def _get_cadvisor_memory_stats(self, container):
        try:
            # Get container stats from cAdvisor
            response = requests.get(f"{Config.cadvisor_endpoint}/{container.id}", timeout=15)
            if not response.ok:
                self._log(f"cAdvisor request failed with status {response.status_code}")
                return None, None

            data = response.json()
            container_data = next(iter(data.values())) # assuming the 1st value in dict will be the cdata
            if not container_data:
                self._log(f"Data not found for {container.name}, id: {container.id} in cAdvisor data")
                return None, None

            stats = container_data.get('stats', [])

            # Get the latest stats entry (should be the most recent)
            latest_stat = stats[-1] if stats else None
            if not latest_stat:
                self._log(f"No stats found for container {container.name}, id: {container.id}")
                return None, None

            memory_stats = latest_stat.get('memory', {})
            return memory_stats.get('usage'), memory_stats.get('max_usage')
        except Exception as e:
            self._log(f"Error fetching cAdvisor data for {container.name}, id: {container.id}: {e}")

        return None, None

    def _calculate_memory_average(self, container_name, current_memory, max_memory):
        # Calculate average of current and max memory
        session_average = (current_memory + max_memory) // 2
        final_average = session_average

        # Get previous average if it exists
        if previous_average := self.mem_stats.get(container_name):
            final_average = (session_average + previous_average) // 2

        return final_average

    def _update_container_memory_usage(self, container):
        """Get container memory usage statistics and update mem_stats."""
        current_memory, max_memory = self._get_cadvisor_memory_stats(container)
        if not current_memory:
            # fallback: Get current memory usage from container API
            current_memory = self._get_current_container_memory(container)

        if not current_memory or current_memory <= 0:
            return

        # Calculate averaged memory usage
        max_memory = max_memory or current_memory
        averaged_memory = self._calculate_memory_average(container.name, current_memory, max_memory)
        self.mem_stats[container.name] = averaged_memory

    def _process_container(self, container):
        """Process a single container for potential stopping."""
        try:
            self._update_container_memory_usage(container)

            if not self._is_container_active(container, Config.min_uptime_hours, Config.inactive_threshold_hours):
                with FileLock(f"/tmp/{container.name}.lock", timeout=0):
                    # stop bench services gracefully and then stop the container
                    Bench(container.name, self.server).docker_execute("supervisorctl stop all", as_root=True)
                    container.stop()
                    self._log(f"Stopped inactive container: {container.name}")

        except docker.errors.NotFound:
            self._log(f"Container {container.name} not found when trying to stop")
        except Timeout:
            self._log(f"Could not acquire lock for {container.name}, skipping")
        except Exception as e:
            self._log(f"Unexpected error processing container {container.name}: {e}")

    def _save_container_stats(self) -> None:
        if not self.mem_stats:
            return

        with FileLock("/tmp/bench_mem_stats.lock"):
            with open(Config.stats_file, 'w') as f:
                json.dump(self.mem_stats, f, indent=2)

        # clear any residue
        self.mem_stats = None

    def start(self):
        retries = 3
        self.running = True
        while self.running:
            try:
                self.sleeping = True
                time.sleep(Config.check_interval_minutes * 60)
                self.sleeping = False

                self._log("Starting Bench Stopper")

                self.server = Server()
                self._init_docker_client()

                running_containers = self.docker_client.containers.list()
                if running_containers:
                    self.mem_stats = self._load_memory_stats(Config.stats_file)

                for container in running_containers:
                    if not self.running:
                        break

                    self._process_container(container)

                self._save_container_stats()
                retries = 3
            except Exception as e:
                self._log(f"Unexpected error in main loop: {e}")

                retries -= 1
                if retries <= 0:
                    self._log("Unable to recover - Exiting")
                    break

        self._log("Bench stopper stopped")


if __name__ == "__main__":
    BenchStopper().start()
