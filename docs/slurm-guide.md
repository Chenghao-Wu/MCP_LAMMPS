# SLURM Integration Guide

This guide explains how to use the SLURM integration features in MCP LAMMPS to run simulations on HPC clusters.

## Overview

MCP LAMMPS supports optional SLURM (Simple Linux Utility for Resource Management) integration, allowing you to submit and manage LAMMPS simulations on HPC clusters. The integration provides:

- **Job Submission**: Submit simulations to SLURM queue with custom resource requirements
- **Status Monitoring**: Check job status and track progress
- **Job Management**: Cancel jobs and retrieve results
- **Flexible Configuration**: Global defaults with per-simulation overrides

## Prerequisites

- SLURM workload manager installed and accessible
- LAMMPS installed on compute nodes
- Shared filesystem between submission and compute nodes
- Required SLURM commands available: `sbatch`, `squeue`, `scancel`, `sacct`

## Configuration

### Default Configuration File

Create or modify `slurm_config.yaml` in your workspace directory:

```yaml
global_defaults:
  partition: "compute"           # SLURM partition
  nodes: 1                        # Number of nodes
  ntasks_per_node: 4             # Tasks per node
  time_limit: "24:00:00"         # Time limit (HH:MM:SS)
  memory_per_cpu: "4G"           # Memory per CPU
  lammps_module: null            # Module load command (optional)
  lammps_executable: "lmp"       # LAMMPS executable
  mpi_command: "mpirun -np {ntasks}"  # MPI command template
  additional_modules: []         # Additional modules to load
  mail_type: null                # Email notifications (optional)
  mail_user: null                # Email address (optional)
```

### Per-Simulation Overrides

You can override defaults for specific simulations:

```yaml
per_simulation_overrides:
  "large_simulation_id":
    nodes: 4
    ntasks_per_node: 16
    time_limit: "72:00:00"
    partition: "high-memory"
```

## Usage

### 1. Create a Simulation

First, create a simulation as usual:

```python
# Using organic liquid simulation as example
result = await create_organic_simulation(
    name="ethanol_md",
    molecules=[{"smiles": "CCO", "count": 500, "name": "ethanol"}],
    target_density=0.789,
    temperature=298.15,
    production_steps=1000000
)

simulation_id = result["simulation_id"]
```

### 2. Submit to SLURM

Submit the simulation to SLURM cluster:

```python
# Basic submission (uses config defaults)
result = await submit_simulation_to_slurm(
    simulation_id=simulation_id
)

job_id = result["job_id"]
print(f"Submitted with job ID: {job_id}")
```

### 3. Submit with Custom Resources

Override configuration for this specific job:

```python
result = await submit_simulation_to_slurm(
    simulation_id=simulation_id,
    partition="gpu",
    nodes=2,
    ntasks_per_node=8,
    time_limit="48:00:00",
    memory_per_cpu="8G",
    mail_type="END",
    mail_user="user@example.com"
)
```

### 4. Check Job Status

Monitor job progress:

```python
# Check by simulation ID
status = await get_slurm_job_status(simulation_id=simulation_id)

print(f"State: {status['state']}")
print(f"Elapsed: {status['slurm_status']['elapsed_time']}")
print(f"Nodes: {status['slurm_status']['nodes']}")
```

### 5. List All Jobs

View all managed SLURM jobs:

```python
# List all jobs
result = await list_slurm_jobs(active_only=False)
print(f"Total jobs: {result['total_jobs']}")

# List only active jobs
result = await list_slurm_jobs(active_only=True)
```

### 6. Update Job Statuses

Refresh status for all active jobs:

```python
result = await update_slurm_job_statuses()
print(f"Updated {result['active_jobs']} active jobs")
```

### 7. Retrieve Results

Get output files and results from completed job:

```python
results = await retrieve_slurm_results(simulation_id=simulation_id)

# Access standard output
stdout = results["files"]["stdout"]

# Access standard error
stderr = results["files"]["stderr"]

# List all available files
files = results["files"]["available_files"]

# Access simulation results
sim_results = results["simulation_results"]
```

### 8. Cancel a Job

Cancel a running or pending job:

```python
result = await cancel_slurm_job(simulation_id=simulation_id)
print(result["message"])
```

## SLURM Batch Script

The system automatically generates SLURM batch scripts. Example generated script:

```bash
#!/bin/bash
#SBATCH --job-name=ethanol_md
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/path/to/workdir/slurm-%j.out
#SBATCH --error=/path/to/workdir/slurm-%j.err

# Job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Load modules
module load lammps/2023

# Change to working directory
cd /path/to/workdir || exit 1

# Run LAMMPS
mpirun -np 4 lmp -in simulation.in

# Check exit status
if [ $? -eq 0 ]; then
    echo "LAMMPS completed successfully"
    exit 0
else
    echo "LAMMPS failed"
    exit 1
fi
```

## Job States

SLURM jobs can be in the following states:

- **PENDING**: Job is waiting in queue
- **RUNNING**: Job is currently executing
- **COMPLETED**: Job finished successfully
- **FAILED**: Job failed or encountered an error
- **CANCELLED**: Job was cancelled by user
- **TIMEOUT**: Job exceeded time limit
- **NODE_FAIL**: Job failed due to node failure
- **SUSPENDED**: Job is suspended

## Best Practices

### Resource Allocation

1. **Start Small**: Test with minimal resources first
2. **Scale Up**: Increase resources based on actual needs
3. **Time Limits**: Set realistic time limits with buffer
4. **Memory**: Monitor and adjust memory requirements

### Job Management

1. **Monitor Regularly**: Check job status periodically
2. **Log Files**: Review stdout/stderr for debugging
3. **Checkpoints**: Use LAMMPS restart files for long jobs
4. **Clean Up**: Retrieve results and clean old jobs

### Configuration

1. **Global Defaults**: Set sensible defaults for most jobs
2. **Per-Simulation**: Override only when necessary
3. **Module Loading**: Configure environment modules correctly
4. **Email Notifications**: Use for long-running jobs

## Troubleshooting

### Job Not Starting

- Check partition availability: `sinfo`
- Verify resource limits: `scontrol show partition <partition>`
- Review queue: `squeue -u $USER`

### Job Failing

1. Check error file: `slurm-<jobid>.err`
2. Verify LAMMPS installation on compute nodes
3. Check module loading in batch script
4. Verify input file paths

### Results Not Available

- Ensure job completed: check status
- Verify shared filesystem access
- Check file permissions
- Review output directory

## MCP Tools Reference

### submit_simulation_to_slurm

Submit a simulation to SLURM cluster.

**Parameters:**
- `simulation_id` (required): Simulation ID to submit
- `partition` (optional): SLURM partition
- `nodes` (optional): Number of nodes
- `ntasks_per_node` (optional): Tasks per node
- `time_limit` (optional): Time limit (HH:MM:SS)
- `memory_per_cpu` (optional): Memory per CPU
- `mail_type` (optional): Email notification type
- `mail_user` (optional): Email address

**Returns:** Job submission result with job ID

### get_slurm_job_status

Get current status of a SLURM job.

**Parameters:**
- `simulation_id` (optional): Simulation ID
- `job_id` (optional): SLURM job ID (provide one of the two)

**Returns:** Job status information

### cancel_slurm_job

Cancel a running or pending job.

**Parameters:**
- `simulation_id` (required): Simulation ID

**Returns:** Cancellation result

### retrieve_slurm_results

Retrieve output files and results.

**Parameters:**
- `simulation_id` (required): Simulation ID

**Returns:** Job output and simulation results

### list_slurm_jobs

List all managed SLURM jobs.

**Parameters:**
- `active_only` (optional, default: false): Only list active jobs

**Returns:** List of job information

### update_slurm_job_statuses

Update status for all active jobs.

**Parameters:** None

**Returns:** Update result with count of active jobs

## Examples

### Example 1: Large-Scale Simulation

```python
# Create simulation
sim_result = await create_organic_simulation(
    name="large_liquid",
    molecules=[{"smiles": "CCCCCCCC", "count": 2000}],
    target_density=0.7,
    production_steps=10000000
)

# Submit with high resources
job_result = await submit_simulation_to_slurm(
    simulation_id=sim_result["simulation_id"],
    nodes=4,
    ntasks_per_node=16,
    time_limit="72:00:00",
    partition="high-memory"
)

# Monitor until complete
import asyncio
while True:
    status = await get_slurm_job_status(simulation_id=sim_result["simulation_id"])
    if status["state"] in ["COMPLETED", "FAILED", "CANCELLED"]:
        break
    await asyncio.sleep(300)  # Check every 5 minutes

# Retrieve results
results = await retrieve_slurm_results(simulation_id=sim_result["simulation_id"])
```

### Example 2: Multiple Jobs

```python
# Submit multiple simulations
job_ids = []
for temp in [250, 275, 300, 325, 350]:
    sim = await create_organic_simulation(
        name=f"water_{temp}K",
        molecules=[{"smiles": "O", "count": 1000}],
        temperature=temp,
        production_steps=5000000
    )
    
    job = await submit_simulation_to_slurm(
        simulation_id=sim["simulation_id"],
        time_limit="24:00:00"
    )
    job_ids.append(job["job_id"])

# Monitor all jobs
jobs = await list_slurm_jobs(active_only=True)
print(f"Active jobs: {jobs['total_jobs']}")
```

## Additional Resources

- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- [LAMMPS Manual](https://docs.lammps.org/)
- [MCP LAMMPS User Guide](user-guide.md)
- [API Reference](api-reference.md)

