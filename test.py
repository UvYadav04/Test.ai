from repowise.repowise import Repowise
from repowise.repowise_tools import RepoWiseTools
import asyncio
repo_path = "D:/Web-development/Veda AI/server"
async def index_repo():
   try:
    repowise = Repowise(repo_path)
    # await repowise.index_repo()
    
    # tools = await repowise.get_mcp_tools()
    # for item in tools:
        # print(f"Name : {item['name']}")
        # print(f"Description : {item['description']}")
        # print(f"Input Schema : {item['inputSchema']}")
        # print(f"Output Schema : {item['outputSchema']}")

        
    repowise_tools = RepoWiseTools(repowise)
    output = await repowise_tools.context(targets=["src/services/pdfService.ts"],include=["callers","callees"],compact=False)
    print(output)
    await repowise.close()
   except Exception as e:
    await repowise.close()
    print(f"Error : {e}")

if __name__ == "__main__":
    asyncio.run(index_repo())