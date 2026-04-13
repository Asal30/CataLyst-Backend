# CataLyst Backend - Docker Build Script for Windows
# Usage: .\build.ps1 [dev|prod|clean]

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("build", "dev", "prod", "stop", "clean", "logs", "shell", "test", "help")]
    [string]$Command = "help"
)

# Colors for output
$RED = "Red"
$GREEN = "Green"
$YELLOW = "Yellow"
$BLUE = "Blue"
$NC = "White" # Default color

function Write-ColorOutput {
    param(
        [string]$Color,
        [string]$Level,
        [string]$Message
    )
    Write-Host "[$Level] $Message" -ForegroundColor $Color
}

function Write-Info {
    param([string]$Message)
    Write-ColorOutput $BLUE "INFO" $Message
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput $GREEN "SUCCESS" $Message
}

function Write-Warning {
    param([string]$Message)
    Write-ColorOutput $YELLOW "WARNING" $Message
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput $RED "ERROR" $Message
}

# Check if Docker is installed
function Test-Docker {
    try {
        $null = Get-Command docker -ErrorAction Stop
        $null = Get-Command docker-compose -ErrorAction Stop
    }
    catch {
        Write-Error "Docker or Docker Compose is not installed. Please install Docker Desktop first."
        exit 1
    }
}

# Build the Docker image
function Invoke-BuildImage {
    Write-Info "Building Docker image: catalyst-backend:latest"
    try {
        docker build -t catalyst-backend:latest .
        Write-Success "Docker image built successfully"
    }
    catch {
        Write-Error "Failed to build Docker image: $($_.Exception.Message)"
        exit 1
    }
}

# Run development environment
function Invoke-Dev {
    Write-Info "Starting development environment..."
    try {
        docker-compose up --build
    }
    catch {
        Write-Error "Failed to start development environment: $($_.Exception.Message)"
        exit 1
    }
}

# Run production environment
function Invoke-Prod {
    Write-Info "Starting production environment..."
    try {
        docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d
        Write-Success "Production environment started"
        Write-Info "API available at: http://localhost:8000"
        Write-Info "Nginx available at: http://localhost"
    }
    catch {
        Write-Error "Failed to start production environment: $($_.Exception.Message)"
        exit 1
    }
}

# Stop all containers
function Stop-All {
    Write-Info "Stopping all containers..."
    try {
        docker-compose down 2>$null
        docker-compose -f docker-compose.yml -f docker-compose.prod.yml down 2>$null
        Write-Success "All containers stopped"
    }
    catch {
        Write-Warning "Some containers may not have been stopped cleanly"
    }
}

# Clean up Docker resources
function Invoke-Clean {
    Write-Warning "This will remove all unused Docker resources. Continue? (y/N)"
    $response = Read-Host
    if ($response -match "^([yY][eE][sS]|[yY])$") {
        Write-Info "Cleaning up Docker resources..."
        try {
            docker system prune -f
            docker volume prune -f
            docker image prune -f
            Write-Success "Docker cleanup completed"
        }
        catch {
            Write-Error "Failed to clean up Docker resources: $($_.Exception.Message)"
        }
    }
    else {
        Write-Info "Cleanup cancelled"
    }
}

# Show usage
function Show-Usage {
    Write-Host "CataLyst Backend - Docker Build Script (Windows)" -ForegroundColor $BLUE
    Write-Host ""
    Write-Host "Usage: .\build.ps1 [-Command] <command>"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  build     Build the Docker image"
    Write-Host "  dev       Start development environment"
    Write-Host "  prod      Start production environment"
    Write-Host "  stop      Stop all running containers"
    Write-Host "  clean     Clean up Docker resources"
    Write-Host "  logs      Show container logs"
    Write-Host "  shell     Open shell in running container"
    Write-Host "  test      Run tests in container"
    Write-Host "  help      Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\build.ps1 -Command build    # Build the image"
    Write-Host "  .\build.ps1 -Command dev      # Start development"
    Write-Host "  .\build.ps1 -Command prod     # Start production"
    Write-Host "  .\build.ps1 -Command stop     # Stop all containers"
}

# Show logs
function Show-Logs {
    Write-Info "Showing container logs..."
    try {
        docker-compose logs -f
    }
    catch {
        Write-Error "Failed to show logs: $($_.Exception.Message)"
    }
}

# Open shell in container
function Enter-Shell {
    Write-Info "Opening shell in container..."
    try {
        docker-compose exec catalyst-api powershell 2>$null || docker-compose exec catalyst-api bash 2>$null || docker-compose exec catalyst-api sh
    }
    catch {
        Write-Error "Failed to open shell in container: $($_.Exception.Message)"
    }
}

# Run tests
function Invoke-Tests {
    Write-Info "Running tests in container..."
    try {
        docker-compose exec catalyst-api python -m pytest --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            docker-compose exec catalyst-api python -m pytest
        }
        else {
            Write-Warning "pytest not installed in container"
        }
    }
    catch {
        Write-Error "Failed to run tests: $($_.Exception.Message)"
    }
}

# Main script logic
Test-Docker

switch ($Command) {
    "build" { Invoke-BuildImage }
    "dev" { Invoke-Dev }
    "prod" { Invoke-Prod }
    "stop" { Stop-All }
    "clean" { Invoke-Clean }
    "logs" { Show-Logs }
    "shell" { Enter-Shell }
    "test" { Invoke-Tests }
    "help" { Show-Usage }
    default {
        Write-Error "Unknown command: $Command"
        Write-Host ""
        Show-Usage
        exit 1
    }
}