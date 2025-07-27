# Configuration Guide

This guide explains how to configure MCP LAMMPS for optimal performance and customization.

## Environment Variables

MCP LAMMPS can be configured using environment variables. These can be set in your shell profile or before running the server.

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_LAMMPS_HOST` | `localhost` | Server host address |
| `MCP_LAMMPS_PORT` | `8000` | Server port number |
| `MCP_LAMMPS_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `MCP_LAMMPS_WORK_DIR` | `./workspace` | Working directory for simulations |
| `MCP_LAMMPS_MAX_SIMULATIONS` | `10` | Maximum number of concurrent simulations |
| `MCP_LAMMPS_TIMEOUT` | `300` | Default timeout in seconds |

### Simulation Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_LAMMPS_DEFAULT_TEMPERATURE` | `300` | Default temperature in Kelvin |
| `MCP_LAMMPS_DEFAULT_PRESSURE` | `1.0` | Default pressure in atm |
| `MCP_LAMMPS_DEFAULT_TIMESTEP` | `0.001` | Default timestep in ps |
| `MCP_LAMMPS_DEFAULT_FORCE_FIELD` | `lj/cut` | Default force field |
| `MCP_LAMMPS_EQUILIBRATION_STEPS` | `1000` | Default equilibration steps |
| `MCP_LAMMPS_PRODUCTION_STEPS` | `10000` | Default production steps |

### Performance Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_LAMMPS_MEMORY_LIMIT` | `4GB` | Memory limit per simulation |
| `MCP_LAMMPS_CPU_LIMIT` | `4` | CPU cores per simulation |
| `MCP_LAMMPS_DISK_LIMIT` | `1GB` | Disk space limit per simulation |
| `MCP_LAMMPS_BACKUP_ENABLED` | `true` | Enable automatic backups |
| `MCP_LAMMPS_BACKUP_INTERVAL` | `3600` | Backup interval in seconds |

### Example Environment Setup

```bash
# Add to ~/.bashrc or ~/.zshrc
export MCP_LAMMPS_HOST=0.0.0.0
export MCP_LAMMPS_PORT=8000
export MCP_LAMMPS_LOG_LEVEL=INFO
export MCP_LAMMPS_WORK_DIR=/home/user/mcp_lammps_workspace
export MCP_LAMMPS_MAX_SIMULATIONS=5
export MCP_LAMMPS_DEFAULT_TEMPERATURE=300
export MCP_LAMMPS_DEFAULT_PRESSURE=1.0
export MCP_LAMMPS_MEMORY_LIMIT=8GB
export MCP_LAMMPS_CPU_LIMIT=8
export MCP_LAMMPS_BACKUP_ENABLED=true
```

## Configuration File

You can also use a YAML configuration file for more detailed settings.

### File Location

The configuration file should be placed at:
- Linux/macOS: `~/.mcp_lammps/config.yaml`
- Windows: `%USERPROFILE%\.mcp_lammps\config.yaml`

### Configuration Structure

```yaml
# Server configuration
server:
  host: localhost
  port: 8000
  log_level: INFO
  timeout: 300
  max_connections: 100
  enable_cors: true
  cors_origins:
    - http://localhost:3000
    - https://your-app.com

# Simulation settings
simulation:
  default_temperature: 300
  default_pressure: 1.0
  default_timestep: 0.001
  default_force_field: lj/cut
  equilibration_steps: 1000
  production_steps: 10000
  max_simulations: 10
  auto_cleanup: true
  cleanup_after_days: 7

# Performance settings
performance:
  memory_limit: 4GB
  cpu_limit: 4
  disk_limit: 1GB
  parallel_simulations: 2
  enable_gpu: false
  gpu_memory_limit: 2GB

# Storage settings
storage:
  work_dir: ./workspace
  backup_enabled: true
  backup_interval: 3600
  backup_retention: 7
  max_file_size: 1GB
  compression_enabled: true
  compression_level: 6

# LAMMPS settings
lammps:
  executable: lmp
  python_interface: true
  packages:
    - python
    - rigid
    - manybody
  kspace_style: pppm
  pair_style: lj/cut
  bond_style: harmonic
  angle_style: harmonic
  dihedral_style: harmonic

# Analysis settings
analysis:
  enable_rmsd: true
  enable_gyration: true
  enable_diffusion: true
  trajectory_skip: 1
  analysis_interval: 1000
  plot_format: png
  plot_dpi: 300

# Logging settings
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/mcp_lammps.log
  max_size: 10MB
  backup_count: 5
  enable_console: true
  enable_file: true

# Security settings
security:
  enable_authentication: false
  api_key_required: false
  allowed_ips:
    - 127.0.0.1
    - ::1
  rate_limit: 100
  rate_limit_window: 3600

# Monitoring settings
monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_interval: 30
  resource_monitoring: true
  alert_on_error: true
  email_alerts: false
  email_recipients: []
```

## Advanced Configuration

### Custom Force Fields

You can define custom force field parameters:

```yaml
lammps:
  custom_force_fields:
    my_water:
      pair_style: lj/cut/coul/long
      pair_coeff: "* * 0.0"
      pair_coeff: "1 1 0.0 0.0"
      pair_coeff: "2 2 0.0 0.0"
      pair_coeff: "1 2 0.0 0.0"
      bond_style: harmonic
      bond_coeff: "1 450.0 0.9572"
      angle_style: harmonic
      angle_coeff: "1 55.0 104.52"
      special_bonds: lj/coul 0.0 0.0 0.5
      kspace_style: pppm 1.0e-4
```

### Simulation Templates

Define reusable simulation templates:

```yaml
simulation:
  templates:
    water_nvt:
      name: "Water NVT Simulation"
      force_field: tip3p
      ensemble: nvt
      temperature: 300
      timestep: 0.001
      equilibration_steps: 1000
      production_steps: 10000
      output_frequency: 100
      
    protein_npt:
      name: "Protein NPT Simulation"
      force_field: charmm
      ensemble: npt
      temperature: 300
      pressure: 1.0
      timestep: 0.002
      equilibration_steps: 5000
      production_steps: 50000
      output_frequency: 500
      
    material_properties:
      name: "Material Properties Calculation"
      force_field: lj/cut
      ensemble: nvt
      temperature: 300
      timestep: 0.001
      equilibration_steps: 2000
      production_steps: 20000
      output_frequency: 200
```

### Output Configuration

Configure output formats and frequencies:

```yaml
output:
  trajectory:
    format: lammpstrj
    frequency: 100
    compression: gzip
    include_velocities: true
    include_forces: false
    
  thermodynamic:
    frequency: 10
    properties:
      - step
      - temp
      - press
      - pe
      - ke
      - etotal
      - volume
      - density
      
  restart:
    frequency: 1000
    format: data
    compression: gzip
    
  analysis:
    frequency: 1000
    properties:
      - rmsd
      - gyration
      - diffusion
      - rdf
```

## Performance Tuning

### Memory Optimization

```yaml
performance:
  memory:
    limit_per_simulation: 2GB
    shared_memory: true
    memory_pool_size: 1GB
    garbage_collection_interval: 1000
    
  cpu:
    cores_per_simulation: 2
    thread_pool_size: 8
    load_balancing: true
    
  gpu:
    enable: false
    memory_limit: 2GB
    precision: mixed
    device_id: 0
```

### Parallel Processing

```yaml
parallel:
  enable: true
  max_workers: 4
  chunk_size: 1000
  load_balancing: true
  shared_memory: true
  
  mpi:
    enable: false
    ranks_per_simulation: 4
    hostfile: hosts.txt
```

## Security Configuration

### Authentication

```yaml
security:
  authentication:
    enabled: true
    method: api_key
    api_keys:
      - name: "user1"
        key: "your-secret-api-key-1"
        permissions: ["read", "write"]
      - name: "user2"
        key: "your-secret-api-key-2"
        permissions: ["read"]
        
  network:
    allowed_hosts:
      - 127.0.0.1
      - 192.168.1.0/24
    ssl_enabled: false
    ssl_cert: /path/to/cert.pem
    ssl_key: /path/to/key.pem
```

## Monitoring and Logging

### Metrics Configuration

```yaml
monitoring:
  metrics:
    enabled: true
    port: 9090
    path: /metrics
    interval: 30
    
  health_check:
    enabled: true
    interval: 30
    timeout: 10
    endpoints:
      - /health
      - /ready
      
  logging:
    level: INFO
    format: json
    output:
      - console
      - file
    file:
      path: logs/mcp_lammps.log
      max_size: 10MB
      backup_count: 5
      rotation: daily
```

## Environment-Specific Configurations

### Development Environment

```yaml
# config-dev.yaml
server:
  host: localhost
  port: 8000
  log_level: DEBUG
  
simulation:
  max_simulations: 2
  auto_cleanup: false
  
performance:
  memory_limit: 1GB
  cpu_limit: 2
  
logging:
  level: DEBUG
  enable_console: true
```

### Production Environment

```yaml
# config-prod.yaml
server:
  host: 0.0.0.0
  port: 8000
  log_level: INFO
  
simulation:
  max_simulations: 20
  auto_cleanup: true
  
performance:
  memory_limit: 8GB
  cpu_limit: 8
  
security:
  enable_authentication: true
  api_key_required: true
  
monitoring:
  enable_metrics: true
  alert_on_error: true
```

### High-Performance Computing (HPC)

```yaml
# config-hpc.yaml
server:
  host: 0.0.0.0
  port: 8000
  
simulation:
  max_simulations: 50
  
performance:
  memory_limit: 32GB
  cpu_limit: 16
  enable_gpu: true
  gpu_memory_limit: 8GB
  
parallel:
  enable: true
  max_workers: 16
  mpi:
    enable: true
    ranks_per_simulation: 8
```

## Configuration Validation

MCP LAMMPS validates configuration files on startup. Common validation errors:

### Invalid Configuration

```yaml
# This will cause validation errors
simulation:
  max_simulations: -1  # Must be positive
  default_temperature: -50  # Must be positive
  
performance:
  memory_limit: invalid  # Must be valid size string
  cpu_limit: 0  # Must be positive
```

### Valid Configuration

```yaml
# This is valid
simulation:
  max_simulations: 10
  default_temperature: 300
  
performance:
  memory_limit: 4GB
  cpu_limit: 4
```

## Configuration Loading Order

MCP LAMMPS loads configuration in the following order:

1. **Default values** - Built-in defaults
2. **Environment variables** - Override defaults
3. **Configuration file** - Override environment variables
4. **Command line arguments** - Override configuration file

### Example Loading

```bash
# Start with custom configuration
MCP_LAMMPS_CONFIG_FILE=/path/to/custom.yaml python -m mcp_lammps.server

# Override specific settings
MCP_LAMMPS_LOG_LEVEL=DEBUG MCP_LAMMPS_PORT=9000 python -m mcp_lammps.server
```

## Troubleshooting Configuration

### Common Issues

1. **Configuration file not found**
   ```bash
   # Check if file exists
   ls -la ~/.mcp_lammps/config.yaml
   
   # Create directory if needed
   mkdir -p ~/.mcp_lammps
   ```

2. **Invalid YAML syntax**
   ```bash
   # Validate YAML syntax
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

3. **Permission denied**
   ```bash
   # Fix permissions
   chmod 644 ~/.mcp_lammps/config.yaml
   ```

4. **Configuration not applied**
   ```bash
   # Check environment variables
   env | grep MCP_LAMMPS
   
   # Restart server after configuration changes
   ```

### Debug Configuration

Enable debug logging to see configuration loading:

```bash
MCP_LAMMPS_LOG_LEVEL=DEBUG python -m mcp_lammps.server
```

This will show:
- Configuration file location
- Loaded settings
- Validation results
- Any configuration errors

## Best Practices

1. **Use environment variables** for sensitive information
2. **Use configuration files** for complex settings
3. **Validate configurations** before deployment
4. **Use different configs** for different environments
5. **Monitor resource usage** and adjust limits accordingly
6. **Backup configurations** along with your data
7. **Document custom settings** for team members

For more information about specific configuration options, see the [API Reference](api-reference.md). 