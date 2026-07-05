import os
import json
import shutil
import asyncio
from repowise.repowise_config import RepoWiseConfig
from dotenv import load_dotenv
load_dotenv()

class Repowise:
    def __init__(self, repo_path: str):
        self.configManager = RepoWiseConfig()
        self.repo_path = repo_path
        self.mcp_process = None
        self.msg_id = 0
        self.config = self.configManager.get_current()

    async def index_repo(self):
        #deleting previous repowise folder so that it doesnot get included in indexing
        self._delete_repowise_folder()
        retries = self.configManager.get_retries()
        for attempt in range(retries):
            try:
                await self._run_index()
                return

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                self._delete_repowise_folder()

        raise RuntimeError("Failed to index repository after 3 attempts.")

    def _delete_repowise_folder(self):
        data_dir = os.path.join(self.repo_path, ".repowise")
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            print(f"Cleaned up {data_dir}")


    async def _run_index(self):
        print(f"Indexing {self.repo_path}...")
        if not os.path.exists(self.repo_path):
            raise FileNotFoundError(f"Path {self.repo_path} not found")
        args = self.__build_args(self.config)
        env = os.environ.copy()

        process = await asyncio.create_subprocess_exec(
            "repowise",
            "init",
            self.repo_path,
            *args,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        print(stdout.decode())
        print(stderr.decode())

        print("Exit code:", process.returncode)


        if process.returncode != 0:
            raise Exception(stderr.decode().strip() or "Indexing failed")

        print("Indexing complete.")

    def __build_args(self, config = {}):
        args = []

        for key, value in config.items():
            flag = "--" + key.replace("_", "-")

            if isinstance(value, bool):
                if value:
                    args.append(flag)
            elif isinstance(value, list):
                for item in value:
                    args.extend([flag, item])
            else:
                args.extend([flag, str(value)])

        return args

    async def __initiate_mcp_server(self):
        # start repowise mcp if not already running
        if self.mcp_process and self.mcp_process.returncode is None:
            return

        self.mcp_process = await asyncio.create_subprocess_exec(
            "repowise", "mcp", self.repo_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.msg_id = 0

        # mcp initialize handshake
        await self.__send_msg("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "repowise-client", "version": "1.0"}
        })

        # send initialized notification (no id, not a request)
        note = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        self.mcp_process.stdin.write(note.encode())
        await self.mcp_process.stdin.drain()

    async def __send_msg(self, method, params=None):
        # send json-rpc message and return matching response
        self.msg_id += 1
        msg = {"jsonrpc": "2.0", "id": self.msg_id, "method": method}
        if params is not None:
            msg["params"] = params

        self.mcp_process.stdin.write((json.dumps(msg) + "\n").encode())
        await self.mcp_process.stdin.drain()

        # skip notifications, wait for response with our id
        while True:
            line = await self.mcp_process.stdout.readline()
            if not line:
                return None
            response = json.loads(line.decode())
            if response.get("id") == self.msg_id:
                return response

    async def close(self):
        if self.mcp_process and self.mcp_process.returncode is None:
            self.mcp_process.terminate()
            await self.mcp_process.wait()

    async def get_mcp_tools(self):
        # list all available mcp tools
        await self.__initiate_mcp_server()
        result = await self.__send_msg("tools/list")
        if result and "result" in result:
            return result["result"].get("tools", [])
        return []

    async def call_mcp_tool(self, tool_name: str, args: dict = {}):
        # call a specific mcp tool
        await self.__initiate_mcp_server()
        result = await self.__send_msg("tools/call", {"name": tool_name, "arguments": args})
        # if result and "result" in result:
        #     return result["result"]
        # return None
        return result

    async def update_repo(self):
        # re-index the repo
        await self.index_repo()

    def delete_repo(self):
        # remove .repowise data folder
        data_dir = os.path.join(self.repo_path, ".repowise")
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            print(f"Removed {data_dir}")
        else:
            print("No repowise data found")

    async def get_repo_info(self):
        # run repowise status and return output
        process = await asyncio.create_subprocess_exec(
            "repowise", "status", self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        return stdout.decode().strip()

    async def stop_mcp(self):
        # kill the running mcp server
        if self.mcp_process and self.mcp_process.returncode is None:
            self.mcp_process.terminate()
            await self.mcp_process.wait()
            self.mcp_process = None