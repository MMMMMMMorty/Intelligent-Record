"""
Engine Manager - GPU-aware model container management
Handles dynamic GPU memory allocation and container lifecycle
"""

import subprocess
import asyncio
import httpx
import docker
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


@dataclass
class GPUInfo:
    """GPU information"""
    available: bool
    name: str
    total_gb: float
    used_gb: float
    free_gb: float
    utilization_percent: float


@dataclass
class EngineStatus:
    """Engine container status"""
    name: str
    status: str  # "offline", "starting", "ready", "stopping", "error"
    container_name: str
    port: int
    model_name: str
    memory_util: float
    started_at: Optional[datetime] = None
    error_message: Optional[str] = None


class EngineManager:
    """
    Manages ASR and LLM containers with GPU-aware memory allocation
    """

    def __init__(self):
        self.engines = {
            "asr": EngineStatus(
                name="asr",
                status="offline",
                container_name="intelligent-record-asr",
                port=8001,
                model_name="Qwen3-ASR-1.7B",
                memory_util=0.0
            ),
            "llm": EngineStatus(
                name="llm",
                status="offline",
                container_name="intelligent-record-llm",
                port=8002,
                model_name="Qwen3-1.7B",
                memory_util=0.0
            )
        }
        self._gpu_info: Optional[GPUInfo] = None
        self._last_gpu_check: Optional[datetime] = None

        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            print(f"Failed to initialize Docker client: {e}")
            self.docker_client = None

    def get_gpu_info(self) -> GPUInfo:
        """Get GPU information from nvidia-smi"""
        # Cache GPU info for 5 seconds
        if (self._last_gpu_check and
            datetime.now() - self._last_gpu_check < timedelta(seconds=5)):
            return self._gpu_info

        try:
            # Query GPU memory info
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=5
            )

            lines = result.stdout.strip().split('\n')
            if lines:
                parts = [p.strip() for p in lines[0].split(',')]
                if len(parts) >= 5:
                    total_mb = float(parts[1])
                    used_mb = float(parts[2])
                    free_mb = float(parts[3])
                    util = float(parts[4])

                    self._gpu_info = GPUInfo(
                        available=True,
                        name=parts[0],
                        total_gb=round(total_mb / 1024, 1),
                        used_gb=round(used_mb / 1024, 1),
                        free_gb=round(free_mb / 1024, 1),
                        utilization_percent=util
                    )
                    self._last_gpu_check = datetime.now()
                    return self._gpu_info

        except Exception as e:
            print(f"Failed to get GPU info: {e}")

        self._gpu_info = GPUInfo(
            available=False, name="Unknown", total_gb=0, used_gb=0, free_gb=0, utilization_percent=0
        )
        return self._gpu_info

    def calculate_memory_allocation(self, total_gb: float) -> Dict[str, float]:
        """
        VRAM allocation for mutual exclusive engines (ASR and LLM don't run simultaneously).
        Each engine gets dedicated 6GB allocation since they never run together.

        For RTX 5060 Ti (8GB) and 4080S (16GB):
        - ASR: 0.75 (~6GB on 8GB card, ~12GB on 16GB card)
        - LLM: 0.75 (~6GB on 8GB card, ~12GB on 16GB card)
        - Engines are mutually exclusive, so total usage never exceeds ~6GB at a time
        """
        if total_gb <= 0:
            total_gb = 8.0

        # Target ~6GB for each engine (mutual exclusive, so no overlap)
        # For 8GB GPU: 6/8 = 0.75
        # For 16GB GPU: 6/16 = 0.375
        target_gb = 6.0
        util = target_gb / total_gb

        # Clamp to safe range
        util = max(0.3, min(0.85, util))

        return {
            "asr": round(util, 3),
            "llm": round(util, 3),
            "target_gb": target_gb,
            "gpu_total_gb": total_gb
        }

    def get_engine_gpu_memory(self, engine_name: str) -> float:
        """
        Get actual GPU memory usage for a specific engine container
        Returns memory used in GB
        """
        engine = self.engines.get(engine_name)
        if not engine or engine.status != "ready":
            return 0.0

        try:
            # Try to get PID of process in container and query its GPU memory
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=5
            )

            total_used_mb = 0
            for line in result.stdout.strip().split('\n'):
                if line.strip() and ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            mem_mb = float(parts[1].strip())
                            total_used_mb += mem_mb
                        except ValueError:
                            continue

            # Convert to GB
            return round(total_used_mb / 1024, 1)

        except Exception as e:
            print(f"Failed to get GPU memory for {engine_name}: {e}")
            return 0.0

    async def check_engine_health(self, engine_name: str) -> bool:
        """Check if engine API is responding"""
        engine = self.engines.get(engine_name)
        if not engine:
            return False

        try:
            async with httpx.AsyncClient() as client:
                # ASR streaming service uses different endpoint
                if engine_name == "asr":
                    # Check root endpoint for streaming ASR
                    response = await client.get(
                        f"http://host.docker.internal:{engine.port}/",
                        timeout=3.0
                    )
                    return response.status_code in [200, 404]
                else:
                    # LLM uses OpenAI API endpoint
                    response = await client.get(
                        f"http://host.docker.internal:{engine.port}/v1/models",
                        timeout=3.0
                    )
                    return response.status_code == 200
        except:
            return False

    async def update_engine_status(self, engine_name: str):
        """Update engine status by checking container and API"""
        engine = self.engines.get(engine_name)
        if not engine:
            return

        if not self.docker_client:
            engine.status = "error"
            engine.error_message = "Docker client not available"
            return

        try:
            # Check if container exists and is running
            try:
                container = self.docker_client.containers.get(engine.container_name)
                if container.status == "running":
                    # Container is running, check API health
                    if await self.check_engine_health(engine_name):
                        engine.status = "ready"
                        if not engine.started_at:
                            engine.started_at = datetime.now()
                    else:
                        # Only set to "starting" if not already "ready"
                        # This prevents race conditions where health check temporarily fails
                        # but engine was previously marked as ready
                        if engine.status != "ready":
                            engine.status = "starting"
                else:
                    # Container exists but not running
                    engine.status = "offline"
                    engine.started_at = None
            except docker.errors.NotFound:
                # Container doesn't exist
                engine.status = "offline"
                engine.started_at = None

        except Exception as e:
            engine.status = "error"
            engine.error_message = str(e)

    async def start_engine(self, engine_name: str) -> Tuple[bool, str]:
        """Start an engine container"""
        engine = self.engines.get(engine_name)
        if not engine:
            return False, f"Unknown engine: {engine_name}"

        if not self.docker_client:
            return False, "Docker client not available"

        # Mutual exclusion: check if other engine is running
        other_engine_name = "llm" if engine_name == "asr" else "asr"
        other_engine = self.engines.get(other_engine_name)
        await self.update_engine_status(other_engine_name)
        if other_engine and other_engine.status == "ready":
            return False, f"Cannot start {engine_name.upper()} engine while {other_engine_name.upper()} is running. Please stop {other_engine_name.upper()} first."

        # Get GPU info and calculate allocation
        gpu = self.get_gpu_info()
        # Use detected GPU memory or default to 8GB
        gpu_total_gb = gpu.total_gb if gpu.available else 0
        allocation = self.calculate_memory_allocation(gpu_total_gb)
        engine.memory_util = allocation[engine_name]

        print(f"[DEBUG] GPU: {gpu.name}, Total: {gpu_total_gb}GB, Target: {allocation.get('target_gb', 6)}GB, Actual util: {allocation.get('actual_utilization', allocation['asr'])}")

        # Check if already running
        await self.update_engine_status(engine_name)
        if engine.status == "ready":
            return True, "Engine already running"

        engine.status = "starting"
        engine.error_message = None

        try:
            # Check if container already exists
            try:
                existing = self.docker_client.containers.get(engine.container_name)
                if existing.status == "exited":
                    # Container exists but stopped, just start it
                    existing.start()
                elif existing.status != "running":
                    # Remove unhealthy container and recreate
                    existing.remove(force=True)
                    if engine_name == "asr":
                        self._run_asr_container(engine, allocation["asr"])
                    else:
                        self._run_llm_container(engine, allocation["llm"])
                # If running, health check will handle it
            except docker.errors.NotFound:
                # No container exists, create new one
                if engine_name == "asr":
                    self._run_asr_container(engine, allocation["asr"])
                else:
                    self._run_llm_container(engine, allocation["llm"])
            except Exception as e:
                # Handle other errors (permission, etc)
                print(f"Container check error: {e}")
                # Fallback: try to create new container
                try:
                    if engine_name == "asr":
                        self._run_asr_container(engine, allocation["asr"])
                    else:
                        self._run_llm_container(engine, allocation["llm"])
                except Exception as inner_e:
                    raise inner_e

            # Wait for service to be ready (up to 120 seconds)
            for i in range(120):
                await asyncio.sleep(1)
                if await self.check_engine_health(engine_name):
                    engine.status = "ready"
                    engine.started_at = datetime.now()
                    return True, "Engine started successfully"

            engine.status = "error"
            engine.error_message = "Timeout waiting for service to be ready"
            return False, "Timeout waiting for service"

        except Exception as e:
            engine.status = "error"
            engine.error_message = str(e)
            return False, str(e)

    async def stop_engine(self, engine_name: str) -> Tuple[bool, str]:
        """Stop an engine container"""
        engine = self.engines.get(engine_name)
        if not engine:
            return False, f"Unknown engine: {engine_name}"

        if not self.docker_client:
            return False, "Docker client not available"

        engine.status = "stopping"

        try:
            try:
                container = self.docker_client.containers.get(engine.container_name)
                container.stop(timeout=10)
                # Do NOT remove - keep container for reuse
                # container.remove(force=True)
            except docker.errors.NotFound:
                pass  # Container does not exist

            engine.status = "offline"
            engine.started_at = None
            return True, "Engine stopped (container preserved for reuse)"

        except Exception as e:
            engine.status = "error"
            engine.error_message = str(e)
            return False, str(e)

    def _run_asr_container(self, engine: EngineStatus, memory_util: float):
        """Run ASR container using docker-py

        Optimized for 6GB VRAM on 8GB GPUs (RTX 5060 Ti / 4080S testing):
        - gpu_memory_utilization=0.75 -> ~6GB on 8GB card
        - max_model_len=256 -> Sufficient for streaming ASR
        - max_num_seqs=1 -> Streaming is single sequence
        """
        return self.docker_client.containers.run(
            image="intelligent-record-asr:latest",  # Custom ASR image with streaming server
            name=engine.container_name,
            detach=True,
            ports={"8000/tcp": engine.port},
            volumes={"/d/Codes/Intelligent-Record/models": {"bind": "/models", "mode": "ro"}},
            environment={
                "NVIDIA_VISIBLE_DEVICES": "all",
                "CUDA_MODULE_LOADING": "LAZY",
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
                "MSYS_NO_PATHCONV": "1"  # Prevent Git Bash path conversion
            },
            device_requests=[
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ],
            command=[
                "python3", "/app/asr_streaming_server.py",
                "--asr-model-path=/models/Qwen3-ASR-1.7B",
                "--host=0.0.0.0",
                "--port=8000",
                f"--gpu-memory-utilization={memory_util}",
                "--max-model-len=256",  # Reduced for 8GB GPU compatibility
                "--max-num-seqs=1"  # Streaming ASR is single sequence
            ]
        )

    def _run_llm_container(self, engine: EngineStatus, memory_util: float):
        """Run LLM container using docker-py

        Using Qwen3-1.7B model for document generation.
        1.7B model fits comfortably in 8GB VRAM without aggressive quantization.

        Settings for RTX 5060 Ti 8GB:
        - max_model_len=2048 (enough for document processing)
        - gpu_memory_utilization ~0.6 (5GB target, safe for 8GB card)
        """
        return self.docker_client.containers.run(
            image="vllm/vllm-openai:nightly",
            name=engine.container_name,
            detach=True,
            ports={"8000/tcp": engine.port},
            volumes={"/d/Codes/Intelligent-Record/models": {"bind": "/models", "mode": "ro"}},
            environment={
                "NVIDIA_VISIBLE_DEVICES": "all",
                "CUDA_MODULE_LOADING": "LAZY",
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
                "MSYS_NO_PATHCONV": "1"
            },
            device_requests=[
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ],
            command=[
                "/models/Qwen3-1.7B",
                "--served-model-name", "qwen3-1.7b",
                "--max-model-len", "2048",  # Increased for document processing
                "--gpu-memory-utilization", str(memory_util),
                "--max-num-seqs", "1",
                "--max-num-batched-tokens", "512",
                "--trust-remote-code",
                "--enforce-eager",
                "--host", "0.0.0.0",
                "--port", "8000"
            ]
        )

    async def get_status_async(self) -> Dict:
        """Get full status of all engines and GPU - async version that refreshes engine status"""
        gpu = self.get_gpu_info()
        allocation = self.calculate_memory_allocation(gpu.total_gb) if gpu.available else {}

        # Refresh engine status before returning (to avoid stale data)
        await self.update_engine_status("asr")
        await self.update_engine_status("llm")

        # Get actual GPU memory usage for each running engine
        engine_status = {}
        for name, engine in self.engines.items():
            actual_memory = 0.0
            if engine.status == "ready":
                actual_memory = self.get_engine_gpu_memory(name)

            engine_status[name] = {
                "status": engine.status,
                "port": engine.port,
                "model_name": engine.model_name,
                "memory_util": engine.memory_util,
                "actual_memory_gb": actual_memory,
                "started_at": engine.started_at.isoformat() if engine.started_at else None,
                "error": engine.error_message
            }

        return {
            "gpu": {
                "available": gpu.available,
                "name": gpu.name,
                "total_gb": gpu.total_gb,
                "used_gb": gpu.used_gb,
                "free_gb": gpu.free_gb,
                "utilization_percent": gpu.utilization_percent
            },
            "allocation": allocation,
            "engines": engine_status
        }

    def get_status(self) -> Dict:
        """Get full status of all engines and GPU - sync version (may return stale data)"""
        gpu = self.get_gpu_info()
        allocation = self.calculate_memory_allocation(gpu.total_gb) if gpu.available else {}

        # Get actual GPU memory usage for each running engine
        engine_status = {}
        for name, engine in self.engines.items():
            actual_memory = 0.0
            if engine.status == "ready":
                actual_memory = self.get_engine_gpu_memory(name)

            engine_status[name] = {
                "status": engine.status,
                "port": engine.port,
                "model_name": engine.model_name,
                "memory_util": engine.memory_util,
                "actual_memory_gb": actual_memory,
                "started_at": engine.started_at.isoformat() if engine.started_at else None,
                "error": engine.error_message
            }

        return {
            "gpu": {
                "available": gpu.available,
                "name": gpu.name,
                "total_gb": gpu.total_gb,
                "used_gb": gpu.used_gb,
                "free_gb": gpu.free_gb,
                "utilization_percent": gpu.utilization_percent
            },
            "allocation": allocation,
            "engines": engine_status
        }


# Global instance
engine_manager = EngineManager()
