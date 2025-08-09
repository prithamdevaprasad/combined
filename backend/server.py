from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
import json
import asyncio
import subprocess
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid
from datetime import datetime

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Models
class FileContent(BaseModel):
    path: str
    content: str

class CompileRequest(BaseModel):
    code: str
    board: str
    sketch_path: str

class UploadRequest(BaseModel):
    code: str
    board: str
    port: str
    sketch_path: str

class LibraryRequest(BaseModel):
    library_name: str

class CoreRequest(BaseModel):
    core_name: str

class LibrarySearchRequest(BaseModel):
    query: str = ""

# Arduino CLI wrapper functions
def run_arduino_cli(command: List[str]) -> Dict:
    """Run arduino-cli command and return result"""
    try:
        # Add arduino-cli to PATH
        env = os.environ.copy()
        bin_path = str(ROOT_DIR.parent / 'bin')
        env['PATH'] = f"{bin_path};{env.get('PATH', '')}"
        # Set HOME to a Windows-compatible path
        env['HOME'] = str(ROOT_DIR)
        
        # Use arduino-cli.exe on Windows
        if command[0] == 'arduino-cli' and os.name == 'nt':
            command[0] = 'arduino-cli.exe'
            
        # Log the command being executed
        logger.info(f"Executing command: {' '.join(command)}")
        logger.info(f"Using bin_path: {bin_path}")
        
        # Check if the executable exists
        cli_path = Path(bin_path) / command[0]
        if not cli_path.exists():
            return {
                'success': False,
                'stdout': '',
                'stderr': f"Arduino CLI executable not found at {cli_path}",
                'returncode': -1
            }
        
        # Execute the command with full path
        command[0] = str(cli_path)
        logger.info(f"Full command path: {command[0]}")
        
        # Run with binary output to avoid encoding issues
        result = subprocess.run(
            command,
            capture_output=True,
            text=False,  # Changed to binary mode
            env=env
        )
        
        # Decode stdout and stderr with error handling
        stdout_str = result.stdout.decode('utf-8', errors='replace') if result.stdout else ''
        stderr_str = result.stderr.decode('utf-8', errors='replace') if result.stderr else ''
        
        return {
            'success': result.returncode == 0,
            'stdout': stdout_str,
            'stderr': stderr_str,
            'returncode': result.returncode
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Arduino Code Editor API"}

@api_router.get("/boards")
async def get_boards():
    """Get list of available boards"""
    result = run_arduino_cli(['arduino-cli', 'board', 'listall', '--format', 'json'])
    
    if result['success']:
        try:
            boards = json.loads(result['stdout'])
            return {"success": True, "boards": boards.get('boards', [])}
        except json.JSONDecodeError:
            return {"success": False, "error": "Failed to parse board list"}
    
    return {"success": False, "error": result['stderr']}

@api_router.get("/boards/available")
async def get_available_boards():
    """Get list of all available boards for installation"""
    result = run_arduino_cli(['arduino-cli', 'board', 'listall', '--format', 'json'])
    
    if result['success']:
        try:
            boards = json.loads(result['stdout'])
            return {"success": True, "boards": boards.get('boards', [])}
        except json.JSONDecodeError:
            return {"success": False, "error": "Failed to parse available boards"}
    
    return {"success": False, "error": result['stderr']}

@api_router.post("/libraries/search")
async def search_libraries(request: LibrarySearchRequest):
    """Search for libraries"""
    if request.query:
        result = run_arduino_cli(['arduino-cli', 'lib', 'search', request.query, '--format', 'json'])
    else:
        result = run_arduino_cli(['arduino-cli', 'lib', 'search', '--format', 'json'])
    
    if result['success']:
        try:
            libraries = json.loads(result['stdout'])
            return {"success": True, "libraries": libraries.get('libraries', [])}
        except json.JSONDecodeError:
            return {"success": False, "error": "Failed to parse library search results"}
    
    return {"success": False, "error": result['stderr']}

@api_router.get("/cores")
async def get_cores():
    """Get list of installed cores"""
    result = run_arduino_cli(['arduino-cli', 'core', 'list', '--format', 'json'])
    
    if result['success']:
        try:
            cores = json.loads(result['stdout'])
            return {"success": True, "cores": cores.get('platforms', [])}
        except json.JSONDecodeError:
            return {"success": False, "error": "Failed to parse cores"}
    
    return {"success": False, "error": result['stderr']}

@api_router.get("/cores/search")
async def search_cores():
    """Get list of all available cores for installation"""
    result = run_arduino_cli(['arduino-cli', 'core', 'search', '--format', 'json'])
    
    if result['success']:
        try:
            cores = json.loads(result['stdout'])
            return {"success": True, "platforms": cores.get('platforms', [])}
        except json.JSONDecodeError:
            return {"success": False, "error": "Failed to parse available cores"}
    
    return {"success": False, "error": result['stderr']}

@api_router.post("/cores/install")
async def install_core(request: CoreRequest):
    """Install a core"""
    result = run_arduino_cli(['arduino-cli', 'core', 'install', request.core_name])
    
    return {
        "success": result['success'],
        "message": result['stdout'] if result['success'] else result['stderr']
    }

@api_router.post("/cores/uninstall")
async def uninstall_core(request: CoreRequest):
    """Uninstall a core"""
    result = run_arduino_cli(['arduino-cli', 'core', 'uninstall', request.core_name])
    
    return {
        "success": result['success'],
        "message": result['stdout'] if result['success'] else result['stderr']
    }

@api_router.get("/ports")
async def get_ports():
    """Get list of available COM ports"""
    result = run_arduino_cli(['arduino-cli', 'board', 'list', '--format', 'json'])
    
    if result['success']:
        try:
            ports = json.loads(result['stdout'])
            return {"success": True, "ports": ports}
        except json.JSONDecodeError:
            return {"success": False, "error": "Failed to parse port list"}
    
    return {"success": False, "error": result['stderr']}

@api_router.get("/libraries")
async def get_libraries():
    """Get list of installed libraries"""
    result = run_arduino_cli(['arduino-cli', 'lib', 'list', '--format', 'json'])
    
    if result['success']:
        try:
            libraries = json.loads(result['stdout'])
            return {"success": True, "libraries": libraries.get('installed_libraries', [])}
        except json.JSONDecodeError:
            return {"success": False, "error": "Failed to parse library list"}
    
    return {"success": False, "error": result['stderr']}

@api_router.post("/libraries/install")
async def install_library(request: LibraryRequest):
    """Install a library"""
    result = run_arduino_cli(['arduino-cli', 'lib', 'install', request.library_name])
    
    return {
        "success": result['success'],
        "message": result['stdout'] if result['success'] else result['stderr']
    }

@api_router.post("/libraries/uninstall")
async def uninstall_library(request: LibraryRequest):
    """Uninstall a library"""
    result = run_arduino_cli(['arduino-cli', 'lib', 'uninstall', request.library_name])
    
    return {
        "success": result['success'],
        "message": result['stdout'] if result['success'] else result['stderr']
    }

@api_router.post("/compile")
async def compile_code(request: CompileRequest):
    """Compile Arduino code"""
    # Create temp directory for sketch
    temp_dir = Path(os.path.join(os.environ.get('TEMP', os.path.join(ROOT_DIR, 'temp')), f"arduino_sketch_{uuid.uuid4()}"))
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    # Write sketch file
    sketch_file = temp_dir / f"{temp_dir.name}.ino"
    sketch_file.write_text(request.code)
    
    # Compile
    result = run_arduino_cli([
        'arduino-cli', 'compile',
        '--fqbn', request.board,
        str(temp_dir)
    ])
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return {
        "success": result['success'],
        "message": result['stdout'] if result['success'] else result['stderr']
    }

@api_router.post("/upload")
async def upload_code(request: UploadRequest):
    """Upload Arduino code to board"""
    # Create temp directory for sketch
    temp_dir = Path(os.path.join(os.environ.get('TEMP', os.path.join(ROOT_DIR, 'temp')), f"arduino_sketch_{uuid.uuid4()}"))
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    # Write sketch file
    sketch_file = temp_dir / f"{temp_dir.name}.ino"
    sketch_file.write_text(request.code)
    
    # Upload
    result = run_arduino_cli([
        'arduino-cli', 'upload',
        '--fqbn', request.board,
        '--port', request.port,
        str(temp_dir)
    ])
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return {
        "success": result['success'],
        "message": result['stdout'] if result['success'] else result['stderr']
    }

@api_router.get("/files/{file_path:path}")
async def get_file(file_path: str):
    """Get file content by path parameter"""
    try:
        # Handle paths that start with /tmp/arduino_workspace
        if file_path.startswith('/tmp/arduino_workspace/'):
            # Replace with the actual workspace directory
            workspace_dir = Path(os.path.join(os.environ.get('TEMP', os.path.join(ROOT_DIR, 'temp')), "arduino_workspace"))
            relative_path = file_path.replace('/tmp/arduino_workspace/', '')
            file_path = workspace_dir / relative_path
        else:
            file_path = Path(file_path)
        if file_path.exists() and file_path.is_file():
            content = file_path.read_text()
            return {"success": True, "content": content}
        else:
            return {"success": False, "error": "File not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@api_router.get("/files")
async def get_file_by_query(path: str):
    """Get file content by query parameter"""
    try:
        logger.info(f"Loading file by query parameter: {path}")
        # Handle paths that start with /tmp/arduino_workspace
        if path.startswith('/tmp/arduino_workspace/'):
            # Replace with the actual workspace directory
            workspace_dir = Path(os.path.join(os.environ.get('TEMP', os.path.join(ROOT_DIR, 'temp')), "arduino_workspace"))
            relative_path = path.replace('/tmp/arduino_workspace/', '')
            file_path = workspace_dir / relative_path
            logger.info(f"Mapped path: {file_path}")
        else:
            file_path = Path(path)
            logger.info(f"Direct path: {file_path}")
        if file_path.exists() and file_path.is_file():
            content = file_path.read_text()
            logger.info(f"File loaded successfully: {file_path}, Size: {len(content)} bytes")
            return {"success": True, "content": content}
        else:
            logger.error(f"File not found: {file_path}")
            return {"success": False, "error": "File not found"}
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@api_router.post("/files")
async def save_file(file_data: FileContent):
    """Save file content"""
    try:
        # Log the incoming request
        logger.info(f"Saving file: {file_data.path}, Content length: {len(file_data.content)}")
        
        # Handle paths that start with /tmp/arduino_workspace
        if file_data.path.startswith('/tmp/arduino_workspace/'):
            # Replace with the actual workspace directory
            workspace_dir = Path(os.path.join(os.environ.get('TEMP', os.path.join(ROOT_DIR, 'temp')), "arduino_workspace"))
            relative_path = file_data.path.replace('/tmp/arduino_workspace/', '')
            file_path = workspace_dir / relative_path
            logger.info(f"Mapped path: {file_path}")
        else:
            file_path = Path(file_data.path)
            logger.info(f"Direct path: {file_path}")
        
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if it's an .ino file
        is_ino_file = file_path.suffix.lower() == '.ino'
        if is_ino_file:
            logger.info(f"Detected .ino file: {file_path}")
        
        # Write the file content
        with open(file_path, 'w', newline='') as f:
            f.write(file_data.content)
        
        # Verify the file was written
        if file_path.exists():
            file_size = file_path.stat().st_size
            logger.info(f"File saved successfully: {file_path}, Size: {file_size} bytes")
            
            # Double-check content for .ino files
            if is_ino_file:
                with open(file_path, 'r') as f:
                    saved_content = f.read()
                if saved_content != file_data.content:
                    logger.error(f"Content mismatch for .ino file: {file_path}")
                    logger.error(f"Expected length: {len(file_data.content)}, Actual length: {len(saved_content)}")
                    # Try again with binary mode
                    with open(file_path, 'wb') as f:
                        f.write(file_data.content.encode('utf-8'))
                    logger.info(f"Retried saving .ino file in binary mode: {file_path}")
            
            return {"success": True, "message": f"File saved successfully. Size: {file_size} bytes"}
        else:
            logger.error(f"File not found after write: {file_path}")
            return {"success": False, "error": "File not found after write operation"}
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

@api_router.delete("/files")
async def delete_file(path: str):
    """Delete a file"""
    try:
        # Handle paths that start with /tmp/arduino_workspace
        if path.startswith('/tmp/arduino_workspace/'):
            # Replace with the actual workspace directory
            workspace_dir = Path(os.path.join(os.environ.get('TEMP', os.path.join(ROOT_DIR, 'temp')), "arduino_workspace"))
            relative_path = path.replace('/tmp/arduino_workspace/', '')
            file_path = workspace_dir / relative_path
        else:
            file_path = Path(path)
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            return {"success": True, "message": "File deleted successfully"}
        else:
            return {"success": False, "error": "File not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@api_router.get("/workspace")
async def get_workspace():
    """Get workspace file tree"""
    workspace_dir = Path(os.path.join(os.environ.get('TEMP', os.path.join(ROOT_DIR, 'temp')), "arduino_workspace"))
    workspace_dir.mkdir(exist_ok=True, parents=True)
    
    def build_tree(path: Path):
        tree = []
        try:
            for item in path.iterdir():
                if item.is_file():
                    tree.append({
                        "name": item.name,
                        "path": str(item),
                        "type": "file"
                    })
                elif item.is_dir():
                    tree.append({
                        "name": item.name,
                        "path": str(item),
                        "type": "directory",
                        "children": build_tree(item)
                    })
        except PermissionError:
            pass
        return tree
    
    return {"success": True, "tree": build_tree(workspace_dir)}

# WebSocket for serial monitor
@app.websocket("/api/serial/{port}")
async def serial_websocket(websocket: WebSocket, port: str):
    await manager.connect(websocket)
    process = None
    try:
        # Start serial monitor
        env = os.environ.copy()
        bin_path = str(ROOT_DIR.parent / 'bin')
        env['PATH'] = f"{bin_path};{env.get('PATH', '')}"
        env['HOME'] = str(ROOT_DIR)
        
        # Use arduino-cli.exe on Windows
        cli_command = 'arduino-cli.exe' if os.name == 'nt' else 'arduino-cli'
        cli_path = Path(bin_path) / cli_command
        
        if not cli_path.exists():
            error_msg = f"Arduino CLI executable not found at {cli_path}"
            logger.error(error_msg)
            await manager.send_personal_message(f"Error: {error_msg}", websocket)
            return
        
        # Get baudrate from query parameters (default to 9600)
        query_params = dict(websocket.query_params)
        baudrate = query_params.get('baudrate', '9600')
        
        # Log connection attempt
        logger.info(f"Attempting to connect to serial port: {port} at {baudrate} baud")
        await manager.send_personal_message(f"Connecting to {port} at {baudrate} baud...", websocket)
        
        # Start the process with stdin pipe for sending data
        try:
            # Configure serial monitor with appropriate settings
            cmd = [
                str(cli_path), 'monitor', 
                '--port', port, 
                '--config', f"baudrate={baudrate}"
                # Removed timeout setting that was causing issues
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                env=env
            )
        except Exception as e:
            error_msg = f"Failed to start arduino-cli monitor: {str(e)}"
            logger.error(error_msg)
            await manager.send_personal_message(f"Error: {error_msg}", websocket)
            return
        
        # Create a task to read from stdout
        async def read_stdout():
            while process and process.poll() is None:
                try:
                    line = process.stdout.readline()
                    if line:
                        await manager.send_personal_message(line.strip(), websocket)
                except Exception as e:
                    logger.error(f"Error reading from serial: {e}")
                    break
        
        # Start the stdout reading task
        read_task = asyncio.create_task(read_stdout())
        
        # Check if process started successfully
        if process.poll() is not None:
            # Process exited immediately
            error_output = process.stderr.read()
            error_msg = f"Failed to connect to port {port}: {error_output}"
            logger.error(error_msg)
            await manager.send_personal_message(f"Error: {error_msg}", websocket)
            return
            
        # Send success message
        await manager.send_personal_message(f"Connected to {port} at {baudrate} baud", websocket)
        
        # Main loop to handle incoming WebSocket messages
        while True:
            if process.poll() is not None:
                error_output = process.stderr.read()
                if error_output:
                    logger.error(f"Process error: {error_output}")
                    await manager.send_personal_message(f"Error: Port monitor error: {error_output}", websocket)
                else:
                    await manager.send_personal_message(f"Serial connection closed", websocket)
                break
                
            # Check for WebSocket messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                # Send data to serial port
                if process and process.poll() is None:
                    try:
                        process.stdin.write(data + '\n')
                        process.stdin.flush()
                        logger.info(f"Sent to serial: {data}")
                    except Exception as e:
                        logger.error(f"Error sending to serial: {e}")
                        await manager.send_personal_message(f"Error sending: {str(e)}", websocket)
                else:
                    await manager.send_personal_message(f"Error: Serial connection is closed", websocket)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for port {port}")
                break
            except Exception as e:
                logger.error(f"Unexpected error in WebSocket loop: {e}")
                await manager.send_personal_message(f"Error: {str(e)}", websocket)
                break
        
        # Cancel the read task
        if read_task and not read_task.done():
            read_task.cancel()
            
    except WebSocketDisconnect:
        logger.info(f"Serial monitor disconnected for port {port}")
    except Exception as e:
        logger.error(f"Serial monitor error: {e}")
        await manager.send_personal_message(f"Error: {str(e)}", websocket)
    finally:
        # Clean up
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=2)
            except Exception as e:
                logger.error(f"Error terminating process: {e}")
                try:
                    process.kill()
                except:
                    pass
        
        manager.disconnect(websocket)
        logger.info(f"Serial monitor connection closed for port {port}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_event():
    # Cleanup resources if needed
    pass