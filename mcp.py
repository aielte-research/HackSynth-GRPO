# Original by DolphinMCP
import os
import sys
import json
import asyncio
from typing import Any, Dict, List, Optional
import logging
logger = logging.getLogger(__name__)
import nest_asyncio, asyncio, json, traceback
nest_asyncio.apply()                   # Jupyter friendliness
_loop = asyncio.get_event_loop()

async def _ensure_started():
    if PYTHON_MCP.process is None:
        ok = await PYTHON_MCP.start()
        if not ok:
            raise RuntimeError("Could not start MCP python-repl")

async def _ensure_and_list():
    await _ensure_started()
    return await PYTHON_MCP.list_tools()

def list_tools_sync():
    try:
        return _loop.run_until_complete(_ensure_and_list())
    except Exception as e:
        traceback.print_exc()
        return []

async def _call_tool(tool_name: str, arguments: dict, timeout: int = 20):
	await _ensure_started()
	res = await PYTHON_MCP.call_tool(tool_name, arguments, timeout)
	if res.get("error") == "Timeout waiting for tool result":
		await PYTHON_MCP.stop()
		## Force the process to be killed
		# os.system("docker exec attackbox pkill -f mcp-python") # OLD method, problem is if multiple ports use MCP kills them all
		port = PYTHON_MCP.port  # assume you stored it on the client
		cmd = (
			f"docker exec attackbox "
			f"bash -c \"pid=$(lsof -t -iTCP:{port} -sTCP:LISTEN) && kill $pid\""
		)
		os.system(cmd)
		ok = await PYTHON_MCP.start()
		if not ok:
			return {"content": [{"type": "text",
				"text": f"Error when restarting MCP Server after timeoit"}],
			"isError": True}
		return {"content": [{"type": "text",
				"text": f"MCP Server timed out."}],
			"isError": True}
	return res

def mcp_call_tool(tool_name: str, arguments: dict, timeout: int = 20) -> str:
    """
    Sync wrapper for execute_python, list_variables, install_package.
    Returns printable string (stdout/result) or '[ERROR] ...'.
    """
    try:
        res = _loop.run_until_complete(_call_tool(tool_name, arguments, timeout))
        if "error" in res:
            return {
            "content": [{"type": "text",
                "text": f"Error when calling tool: {res}"}],
            "isError": True,
            }
        return res
    except asyncio.TimeoutError as e:
        traceback.print_exc()
        return {
            "content": [{"type": "text",
                         "text": f"[EXCEPTION] {str(e)}"}],
            "isError": True,
        }


class MCPClient:
	"""Implementation for a single MCP server."""
	def __init__(self, server_name, command, port, args=None, env=None):
		self.server_name = server_name
		self.command = command
		self.args = args or []
		self.env = env
		self.process = None
		self.tools = []
		self.request_id = 0
		self.protocol_version = "2024-11-05"
		self.port = port
		self.receive_task = None
		self.responses = {}
		self.server_capabilities = {}
		self._shutdown = False
		self._cleanup_lock = asyncio.Lock()

	async def _receive_loop(self):
		if not self.process or self.process.stdout.at_eof():
			return
		try:
			while not self.process.stdout.at_eof():
				line = await self.process.stdout.readline()
				if not line:
					break
				try:
					message = json.loads(line.decode().strip())
					self._process_message(message)
				except json.JSONDecodeError:
					pass
		except Exception:
			pass

	def _process_message(self, message: dict):
		if "jsonrpc" in message and "id" in message:
			if "result" in message or "error" in message:
				self.responses[message["id"]] = message
			else:
				# request from server, not implemented
				resp = {
					"jsonrpc": "2.0",
					"id": message["id"],
					"error": {
						"code": -32601,
						"message": f"Method {message.get('method')} not implemented in client"
					}
				}
				asyncio.create_task(self._send_message(resp))
		elif "jsonrpc" in message and "method" in message and "id" not in message:
			# notification from server
			pass

	async def start(self):
		self._shutdown = False
		expanded_args = []
		for a in self.args:
			if isinstance(a, str) and "~" in a:
				expanded_args.append(os.path.expanduser(a))
			else:
				expanded_args.append(a)

		env_vars = os.environ.copy()
		if self.env:
			env_vars.update(self.env)

		try:
			self.process = await asyncio.create_subprocess_exec(
				self.command,
				*expanded_args,
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=env_vars
			)
			self.receive_task = asyncio.create_task(self._receive_loop())
			return await self._perform_initialize()
		except Exception:
			return False

	async def _perform_initialize(self):
		self.request_id += 1
		req_id = self.request_id
		req = {
			"jsonrpc": "2.0",
			"id": req_id,
			"method": "initialize",
			"params": {
				"protocolVersion": self.protocol_version,
				"capabilities": {"sampling": {}},
				"clientInfo": {
					"name": "DolphinMCPClient",
					"version": "1.0.0"
				}
			}
		}
		await self._send_message(req)

		start = asyncio.get_event_loop().time()
		while asyncio.get_event_loop().time() - start < 5:
			if req_id in self.responses:
				resp = self.responses[req_id]
				del self.responses[req_id]
				if "error" in resp:
					return False
				if "result" in resp:
					note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
					await self._send_message(note)
					init_result = resp["result"]
					self.server_capabilities = init_result.get("capabilities", {})
					return True
			await asyncio.sleep(0.05)
		return False

	async def list_tools(self):
		if not self.process:
			return []
		self.request_id += 1
		rid = self.request_id
		req = {
			"jsonrpc": "2.0",
			"id": rid,
			"method": "tools/list",
			"params": {}
		}
		await self._send_message(req)

		start = asyncio.get_event_loop().time()
		while asyncio.get_event_loop().time() - start < 5:
			if rid in self.responses:
				resp = self.responses[rid]
				del self.responses[rid]
				if "error" in resp:
					return []
				if "result" in resp and "tools" in resp["result"]:
					self.tools = resp["result"]["tools"]
					return self.tools
			await asyncio.sleep(0.05)
		return []

	async def call_tool(self, tool_name: str, arguments: dict, timeout: int = 60):
		if not self.process:
			return {"error": "Not started"}
		self.request_id += 1
		rid = self.request_id
		req = {
			"jsonrpc": "2.0",
			"id": rid,
			"method": "tools/call",
			"params": {
				"name": tool_name,
				"arguments": arguments
			}
		}
		await self._send_message(req)

		start = asyncio.get_event_loop().time()
		while asyncio.get_event_loop().time() - start < timeout:
			if rid in self.responses:
				resp = self.responses[rid]
				del self.responses[rid]
				if "error" in resp:
					return {"error": resp["error"]}
				if "result" in resp:
					return resp["result"]
			await asyncio.sleep(0.05)
		return {"error": "Timeout waiting for tool result"}

	async def _send_message(self, message: dict):
		if not self.process or self._shutdown:
			logger.error(f"Server {self.server_name}: Cannot send message - process not running or shutting down")
			return False
		try:
			data = json.dumps(message) + "\n"
			self.process.stdin.write(data.encode())
			await self.process.stdin.drain()
			return True
		except Exception as e:
			logger.error(f"Server {self.server_name}: Error sending message: {str(e)}")
			return False

	async def stop(self):
		async with self._cleanup_lock:
			if self._shutdown:
				return
			self._shutdown = True

			if self.receive_task and not self.receive_task.done():
				self.receive_task.cancel()
				try:
					await self.receive_task
				except asyncio.CancelledError:
					pass

			if self.process:
				try:
					# Try graceful shutdown first
					self.process.terminate()
					try:
						await asyncio.wait_for(self.process.wait(), timeout=2.0)
					except asyncio.TimeoutError:
						# Force kill if graceful shutdown fails
						logger.warning(f"Server {self.server_name}: Force killing process after timeout")
						self.process.kill()
						await self.process.wait()
				except Exception as e:
					logger.error(f"Server {self.server_name}: Error during process cleanup: {str(e)}")
				finally:
					if self.process.stdin:
						self.process.stdin.close()
					self.process = None

	# Alias close to stop for backward compatibility
	async def close(self):
		await self.stop()

	# Add async context manager support
	async def __aenter__(self):
		await self.start()
		return self

	async def __aexit__(self, exc_type, exc_val, exc_tb):
		await self.stop()



PYTHON_MCP = MCPClient(
    server_name="python-repl",
    command="docker",
	port=8029,
    args=[
        "exec", "-i",
        "attackbox",
        "uv", "--directory", "/opt/mcp-python", "run", "mcp-python",
        "--host", "0.0.0.0", "--port", "8029"
    ],
    env={
        "CODE_STORAGE_DIR": "/tmp/mcp_code",
        "ENV_TYPE": "venv",
        "VENV_PATH": "/opt/mcp-python/venv",
    },
)