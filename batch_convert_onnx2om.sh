
# input op name, for example: Conv_311
op_name=$1

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

# batch convert onnx to om
for file in ${ONNX_PATH}/*;
do
    if [ -f "$file" ]; then
        # get filename without path
        filename=$(basename "$file")     
        name_without_ext="${filename%.*}"
        echo "$name_without_ext"
        # convert using omg
        "${TOOLS_OMG_PATH}/omg" \
            --model "${ONNX_PATH}/${name_without_ext}.onnx" \
            --framework 5 \
            --output "${OM_PATH}/${name_without_ext}"
    fi
done

