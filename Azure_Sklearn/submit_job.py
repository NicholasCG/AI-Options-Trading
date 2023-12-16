# Handle to the workspace
from azure.ai.ml import MLClient

# Authentication package
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

credential = DefaultAzureCredential()

# # Get a handle to the workspace
# ml_client = MLClient(
#     credential=credential,
#     subscription_id="<SUBSCRIPTION_ID>",
#     resource_group_name="<RESOURCE_GROUP>",
#     workspace_name="<AML_WORKSPACE_NAME>",
# )