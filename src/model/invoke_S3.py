import boto3

# Management plane — use "bedrock-agent", NOT "bedrock-agentcore"
client = boto3.client("bedrock-agent", region_name="ap-south-1")

# List agents
agents = client.list_agents()
for a in agents.get("agentSummaries", []):
    print(f"Name: {a['agentName']}  |  ID: {a['agentId']}  |  Status: {a['agentStatus']}")