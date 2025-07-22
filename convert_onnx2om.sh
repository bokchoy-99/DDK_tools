
# input op name
op_name=$1
# input onnx name
onnx_name=$2

# path variables
TOOLS_OMG_PATH=/data/yzl/Workspace/DDK_tools/tools/tools_omg
ONNX_PATH=/data/yzl/Workspace/PatchNet/output/onnx_output/test_npu/onnx/${op_name}
if [ ! -d "$ONNX_PATH" ]; then
    echo "Error: Directory $ONNX_PATH does not exist."
    exit 1
fi

OM_PATH=/data/yzl/Workspace/PatchNet/output/onnx_output/test_npu/om/${op_name}
if [ ! -d "$OM_PATH" ]; then
    mkdir -p "$OM_PATH"
fi

# convert onnx to om
# ./omg --model PatchNet_color.onnx --framework 5 --output ./PatchNet_color
${TOOLS_OMG_PATH}/omg --model ${ONNX_PATH}/${onnx_name}.onnx --framework 5 --output ${OM_PATH}/${onnx_name}

