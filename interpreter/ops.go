package interpreter

// ops.go — TFLite op kernel implementations for MicroWakeWord models.

import (
	"fmt"
	"math"

	"github.com/zserge/microwakeword/model"
)

// opFunc is the signature for all op implementations.
type opFunc func(interp *Interpreter, op model.Operator) error

func getOpDispatch() map[int]opFunc {
	return map[int]opFunc{
		model.OpCallOnce:        opCallOnce,
		model.OpVarHandle:       opVarHandle,
		model.OpReshape:         opReshape,
		model.OpReadVariable:    opReadVariable,
		model.OpAssignVariable:  opAssignVariable,
		model.OpConcatenation:   opConcatenation,
		model.OpStridedSlice:    opStridedSlice,
		model.OpConv2D:          opConv2D,
		model.OpDepthwiseConv2D: opDepthwiseConv2D,
		model.OpFullyConnected:  opFullyConnected,
		model.OpLogistic:        opLogistic,
		model.OpQuantize:        opQuantize,
		model.OpSplitV:          opSplitV,
	}
}

// --- Simple ops ---

func opCallOnce(interp *Interpreter, op model.Operator) error {
	if interp.callOnceExecuted {
		return nil
	}
	idx, ok := op.Options["init_subgraph_index"]
	if !ok {
		return fmt.Errorf("CALL_ONCE: missing init_subgraph_index")
	}
	sgIdx := idx.(int)
	if sgIdx < 0 || sgIdx >= len(interp.model.Subgraphs) {
		return fmt.Errorf("CALL_ONCE: invalid subgraph %d", sgIdx)
	}
	initSG := interp.model.Subgraphs[sgIdx]
	// Allocate tensors for the init subgraph
	initTensors := make([]tensorData, len(initSG.Tensors))
	for i, t := range initSG.Tensors {
		initTensors[i] = makeTensorData(t, interp.model)
	}
	// Execute init subgraph ops
	saved := interp.tensors
	savedSG := interp.sgIdx
	interp.tensors = initTensors
	interp.sgIdx = sgIdx
	dispatch := getOpDispatch()
	for _, initOp := range initSG.Operators {
		fn, ok := dispatch[initOp.OpCode]
		if !ok {
			return fmt.Errorf("CALL_ONCE init: unsupported op %s", model.OpName(initOp.OpCode))
		}
		if err := fn(interp, initOp); err != nil {
			return fmt.Errorf("CALL_ONCE init op %s: %w", model.OpName(initOp.OpCode), err)
		}
	}
	interp.tensors = saved
	interp.sgIdx = savedSG
	interp.callOnceExecuted = true
	return nil
}

func opVarHandle(interp *Interpreter, op model.Operator) error {
	if len(op.Outputs) != 1 {
		return fmt.Errorf("VAR_HANDLE: expected 1 output, got %d", len(op.Outputs))
	}
	// Determine the shared resource name.
	// First try FlexBuffer custom options.
	sharedName := model.ExtractFlexBufferString(op.CustomOptions, "shared_name")
	if sharedName == "" {
		// Fallback: use the tensor name, normalized to strip trailing
		// digits from the last path component.
		// This handles "stream_5/states1" → "stream_5/states".
		sg := interp.model.Subgraphs[interp.sgIdx]
		sharedName = normalizeResourceName(sg.Tensors[op.Outputs[0]].Name)
	}
	outIdx := op.Outputs[0]
	interp.varHandleNames[interp.sgIdx*1000+outIdx] = sharedName
	return nil
}

// normalizeResourceName strips trailing digits from the last path component.
// "stream_5/states1" → "stream_5/states", "stream/states" → "stream/states"
func normalizeResourceName(name string) string {
	// Find the last '/' and strip trailing digits from the suffix.
	lastSlash := -1
	for i := len(name) - 1; i >= 0; i-- {
		if name[i] == '/' {
			lastSlash = i
			break
		}
	}
	suffix := name[lastSlash+1:]
	// Strip trailing digits from suffix
	end := len(suffix)
	for end > 0 && suffix[end-1] >= '0' && suffix[end-1] <= '9' {
		end--
	}
	if end == 0 || end == len(suffix) {
		return name // No digits to strip or all digits
	}
	return name[:lastSlash+1] + suffix[:end]
}

func getResourceName(interp *Interpreter, tensorIdx int) string {
	name, ok := interp.varHandleNames[interp.sgIdx*1000+tensorIdx]
	if ok {
		return name
	}
	// Fallback to tensor name
	sg := interp.model.Subgraphs[interp.sgIdx]
	return sg.Tensors[tensorIdx].Name
}

func opReadVariable(interp *Interpreter, op model.Operator) error {
	if len(op.Inputs) != 1 || len(op.Outputs) != 1 {
		return fmt.Errorf("READ_VARIABLE: expected 1 input 1 output")
	}
	handleName := getResourceName(interp, op.Inputs[0])
	data, ok := interp.resources[handleName]
	if !ok {
		return fmt.Errorf("READ_VARIABLE: resource %q not found", handleName)
	}
	out := &interp.tensors[op.Outputs[0]]
	copy(out.data, data)
	return nil
}

func opAssignVariable(interp *Interpreter, op model.Operator) error {
	if len(op.Inputs) != 2 {
		return fmt.Errorf("ASSIGN_VARIABLE: expected 2 inputs")
	}
	handleName := getResourceName(interp, op.Inputs[0])
	src := interp.tensors[op.Inputs[1]].data
	// Store a copy in the resource variable store
	dst := make([]byte, len(src))
	copy(dst, src)
	interp.resources[handleName] = dst
	return nil
}

func opReshape(interp *Interpreter, op model.Operator) error {
	if len(op.Inputs) < 1 || len(op.Outputs) != 1 {
		return fmt.Errorf("RESHAPE: bad io counts")
	}
	in := &interp.tensors[op.Inputs[0]]
	out := &interp.tensors[op.Outputs[0]]
	// Just copy data; the shape is already set from the tensor definition.
	copy(out.data, in.data)
	return nil
}

func opConcatenation(interp *Interpreter, op model.Operator) error {
	if len(op.Outputs) != 1 {
		return fmt.Errorf("CONCATENATION: bad output count")
	}
	sg := interp.model.Subgraphs[interp.sgIdx]
	axis := 0
	if v, ok := op.Options["axis"]; ok {
		axis = v.(int)
	}
	outTensor := sg.Tensors[op.Outputs[0]]
	out := &interp.tensors[op.Outputs[0]]

	// Compute the concatenation along `axis`.
	outShape := outTensor.Shape
	ndim := len(outShape)
	if axis < 0 {
		axis += ndim
	}

	// outerSize = product of dims before axis
	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= outShape[i]
	}
	// innerSize = product of dims after axis
	innerSize := 1
	for i := axis + 1; i < ndim; i++ {
		innerSize *= outShape[i]
	}

	dstOff := 0
	for outer := 0; outer < outerSize; outer++ {
		for _, inIdx := range op.Inputs {
			inT := sg.Tensors[inIdx]
			inData := interp.tensors[inIdx].data
			axisLen := inT.Shape[axis]
			chunkSize := axisLen * innerSize
			srcOff := outer * chunkSize
			copy(out.data[dstOff:dstOff+chunkSize], inData[srcOff:srcOff+chunkSize])
			dstOff += chunkSize
		}
	}
	return nil
}

func opStridedSlice(interp *Interpreter, op model.Operator) error {
	if len(op.Inputs) < 4 || len(op.Outputs) != 1 {
		return fmt.Errorf("STRIDED_SLICE: bad io")
	}
	sg := interp.model.Subgraphs[interp.sgIdx]
	inShape := sg.Tensors[op.Inputs[0]].Shape
	inData := interp.tensors[op.Inputs[0]].data
	outShape := sg.Tensors[op.Outputs[0]].Shape
	out := &interp.tensors[op.Outputs[0]]

	ndim := len(inShape)
	beginData := interp.tensors[op.Inputs[1]].data
	endData := interp.tensors[op.Inputs[2]].data
	strideData := interp.tensors[op.Inputs[3]].data

	beginMask := 0
	endMask := 0
	if v, ok := op.Options["begin_mask"]; ok {
		beginMask = v.(int)
	}
	if v, ok := op.Options["end_mask"]; ok {
		endMask = v.(int)
	}

	begin := make([]int, ndim)
	end := make([]int, ndim)
	stride := make([]int, ndim)
	for d := 0; d < ndim; d++ {
		begin[d] = getInt32FromBytes(beginData, d)
		end[d] = getInt32FromBytes(endData, d)
		stride[d] = getInt32FromBytes(strideData, d)
		if beginMask&(1<<uint(d)) != 0 {
			begin[d] = 0
		}
		if endMask&(1<<uint(d)) != 0 {
			end[d] = inShape[d]
		}
		if begin[d] < 0 {
			begin[d] += inShape[d]
		}
		if end[d] < 0 {
			end[d] += inShape[d]
		}
		if stride[d] == 0 {
			stride[d] = 1
		}
	}

	// Iterate over output positions and copy from input
	outTotal := model.TensorSize(outShape)
	outCoord := make([]int, ndim)
	for i := 0; i < outTotal; i++ {
		// Compute input offset
		inOff := 0
		inStride := 1
		for d := ndim - 1; d >= 0; d-- {
			inIdx := begin[d] + outCoord[d]*stride[d]
			inOff += inIdx * inStride
			inStride *= inShape[d]
		}
		out.data[i] = inData[inOff]
		// Increment output coordinates
		for d := ndim - 1; d >= 0; d-- {
			outCoord[d]++
			if outCoord[d] < outShape[d] {
				break
			}
			outCoord[d] = 0
		}
	}
	return nil
}

func getInt32FromBytes(data []byte, idx int) int {
	off := idx * 4
	if off+4 > len(data) {
		return 0
	}
	v := int32(data[off]) | int32(data[off+1])<<8 | int32(data[off+2])<<16 | int32(data[off+3])<<24
	return int(v)
}

// opConv2D implements quantized 2D convolution.
// Input: [batch, inH, inW, inC], Filter: [outC, fH, fW, inC], Bias: [outC]
// Output: [batch, outH, outW, outC]
func opConv2D(interp *Interpreter, op model.Operator) error {
	if len(op.Inputs) < 3 || len(op.Outputs) != 1 {
		return fmt.Errorf("CONV_2D: bad io")
	}
	sg := interp.model.Subgraphs[interp.sgIdx]

	inT := sg.Tensors[op.Inputs[0]]
	filterT := sg.Tensors[op.Inputs[1]]
	outT := sg.Tensors[op.Outputs[0]]

	inData := interp.tensors[op.Inputs[0]].data
	filterData := interp.tensors[op.Inputs[1]].data
	biasData := interp.tensors[op.Inputs[2]].data
	outData := interp.tensors[op.Outputs[0]].data

	padding := 0
	strideW := 1
	strideH := 1
	activation := 0
	dilationW := 1
	dilationH := 1
	if v, ok := op.Options["padding"]; ok {
		padding = v.(int)
	}
	if v, ok := op.Options["stride_w"]; ok {
		strideW = v.(int)
	}
	if v, ok := op.Options["stride_h"]; ok {
		strideH = v.(int)
	}
	if v, ok := op.Options["fused_activation"]; ok {
		activation = v.(int)
	}
	if v, ok := op.Options["dilation_w"]; ok {
		dilationW = v.(int)
	}
	if v, ok := op.Options["dilation_h"]; ok {
		dilationH = v.(int)
	}

	batch := inT.Shape[0]
	inH := inT.Shape[1]
	inW := inT.Shape[2]
	inC := inT.Shape[3]
	outC := filterT.Shape[0]
	fH := filterT.Shape[1]
	fW := filterT.Shape[2]
	outH := outT.Shape[1]
	outW := outT.Shape[2]

	inputZP := int32(0)
	if len(inT.Quant.ZeroPoint) > 0 {
		inputZP = int32(inT.Quant.ZeroPoint[0])
	}
	outputZP := int32(0)
	if len(outT.Quant.ZeroPoint) > 0 {
		outputZP = int32(outT.Quant.ZeroPoint[0])
	}

	// Compute per-channel multipliers
	inScale := float64(0)
	if len(inT.Quant.Scale) > 0 {
		inScale = float64(inT.Quant.Scale[0])
	}
	outScale := float64(0)
	if len(outT.Quant.Scale) > 0 {
		outScale = float64(outT.Quant.Scale[0])
	}
	qmuls := make([]QuantMultiplier, outC)
	for c := 0; c < outC; c++ {
		filterScale := float64(0)
		if c < len(filterT.Quant.Scale) {
			filterScale = float64(filterT.Quant.Scale[c])
		} else if len(filterT.Quant.Scale) > 0 {
			filterScale = float64(filterT.Quant.Scale[0])
		}
		if outScale == 0 {
			outScale = 1.0 // avoid division by zero
		}
		qmuls[c] = QuantizeMultiplier(inScale * filterScale / outScale)
	}

	// Compute padding offsets
	padH, padW := 0, 0
	if padding == model.PaddingSame {
		effFH := (fH-1)*dilationH + 1
		effFW := (fW-1)*dilationW + 1
		padH = ((outH-1)*strideH + effFH - inH) / 2
		padW = ((outW-1)*strideW + effFW - inW) / 2
		if padH < 0 {
			padH = 0
		}
		if padW < 0 {
			padW = 0
		}
	}

	// Activation bounds
	actMin, actMax := activationBounds(activation, outputZP)

	for b := 0; b < batch; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for oc := 0; oc < outC; oc++ {
					acc := int32(getBias32(biasData, oc))
					for fh := 0; fh < fH; fh++ {
						for fw := 0; fw < fW; fw++ {
							ih := oh*strideH - padH + fh*dilationH
							iw := ow*strideW - padW + fw*dilationW
							if ih < 0 || ih >= inH || iw < 0 || iw >= inW {
								continue
							}
							for ic := 0; ic < inC; ic++ {
								inOff := ((b*inH+ih)*inW+iw)*inC + ic
								fOff := ((oc*fH+fh)*fW+fw)*inC + ic
								inVal := int32(int8(inData[inOff])) - inputZP
								fVal := int32(int8(filterData[fOff]))
								acc += inVal * fVal
							}
						}
					}
					acc = MultiplyByQuantizedMultiplier(acc, qmuls[oc])
					acc += outputZP
					if acc < actMin {
						acc = actMin
					}
					if acc > actMax {
						acc = actMax
					}
					outOff := ((b*outH+oh)*outW+ow)*outC + oc
					outData[outOff] = byte(int8(ClampInt8(acc)))
				}
			}
		}
	}
	return nil
}

// opDepthwiseConv2D implements quantized depthwise convolution.
// Input: [batch, inH, inW, inC], Filter: [1, fH, fW, outC], Bias: [outC]
// outC = inC * depth_multiplier
func opDepthwiseConv2D(interp *Interpreter, op model.Operator) error {
	if len(op.Inputs) < 3 || len(op.Outputs) != 1 {
		return fmt.Errorf("DEPTHWISE_CONV_2D: bad io")
	}
	sg := interp.model.Subgraphs[interp.sgIdx]

	inT := sg.Tensors[op.Inputs[0]]
	filterT := sg.Tensors[op.Inputs[1]]
	outT := sg.Tensors[op.Outputs[0]]

	inData := interp.tensors[op.Inputs[0]].data
	filterData := interp.tensors[op.Inputs[1]].data
	biasData := interp.tensors[op.Inputs[2]].data
	outData := interp.tensors[op.Outputs[0]].data

	padding := 0
	strideW := 1
	strideH := 1
	depthMul := 1
	activation := 0
	dilationW := 1
	dilationH := 1
	if v, ok := op.Options["padding"]; ok {
		padding = v.(int)
	}
	if v, ok := op.Options["stride_w"]; ok {
		strideW = v.(int)
	}
	if v, ok := op.Options["stride_h"]; ok {
		strideH = v.(int)
	}
	if v, ok := op.Options["depth_multiplier"]; ok {
		depthMul = v.(int)
	}
	if v, ok := op.Options["fused_activation"]; ok {
		activation = v.(int)
	}
	if v, ok := op.Options["dilation_w"]; ok {
		dilationW = v.(int)
	}
	if v, ok := op.Options["dilation_h"]; ok {
		dilationH = v.(int)
	}

	batch := inT.Shape[0]
	inH := inT.Shape[1]
	inW := inT.Shape[2]
	inC := inT.Shape[3]
	fH := filterT.Shape[1]
	fW := filterT.Shape[2]
	outC := inC * depthMul
	outH := outT.Shape[1]
	outW := outT.Shape[2]

	inputZP := int32(0)
	if len(inT.Quant.ZeroPoint) > 0 {
		inputZP = int32(inT.Quant.ZeroPoint[0])
	}
	outputZP := int32(0)
	if len(outT.Quant.ZeroPoint) > 0 {
		outputZP = int32(outT.Quant.ZeroPoint[0])
	}

	inScale := float64(0)
	if len(inT.Quant.Scale) > 0 {
		inScale = float64(inT.Quant.Scale[0])
	}
	outScale := float64(0)
	if len(outT.Quant.Scale) > 0 {
		outScale = float64(outT.Quant.Scale[0])
	}
	qmuls := make([]QuantMultiplier, outC)
	for c := 0; c < outC; c++ {
		filterScale := float64(0)
		if c < len(filterT.Quant.Scale) {
			filterScale = float64(filterT.Quant.Scale[c])
		} else if len(filterT.Quant.Scale) > 0 {
			filterScale = float64(filterT.Quant.Scale[0])
		}
		if outScale == 0 {
			outScale = 1.0
		}
		qmuls[c] = QuantizeMultiplier(inScale * filterScale / outScale)
	}

	padH, padW := 0, 0
	if padding == model.PaddingSame {
		effFH := (fH-1)*dilationH + 1
		effFW := (fW-1)*dilationW + 1
		padH = ((outH-1)*strideH + effFH - inH) / 2
		padW = ((outW-1)*strideW + effFW - inW) / 2
		if padH < 0 {
			padH = 0
		}
		if padW < 0 {
			padW = 0
		}
	}

	actMin, actMax := activationBounds(activation, outputZP)

	for b := 0; b < batch; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for ic := 0; ic < inC; ic++ {
					for dm := 0; dm < depthMul; dm++ {
						oc := ic*depthMul + dm
						acc := int32(getBias32(biasData, oc))
						for fh := 0; fh < fH; fh++ {
							for fw := 0; fw < fW; fw++ {
								ih := oh*strideH - padH + fh*dilationH
								iw := ow*strideW - padW + fw*dilationW
								if ih < 0 || ih >= inH || iw < 0 || iw >= inW {
									continue
								}
								inOff := ((b*inH+ih)*inW+iw)*inC + ic
								fOff := (fh*fW+fw)*outC + oc
								inVal := int32(int8(inData[inOff])) - inputZP
								fVal := int32(int8(filterData[fOff]))
								acc += inVal * fVal
							}
						}
						acc = MultiplyByQuantizedMultiplier(acc, qmuls[oc])
						acc += outputZP
						if acc < actMin {
							acc = actMin
						}
						if acc > actMax {
							acc = actMax
						}
						outOff := ((b*outH+oh)*outW+ow)*outC + oc
						outData[outOff] = byte(int8(ClampInt8(acc)))
					}
				}
			}
		}
	}
	return nil
}

// opFullyConnected implements quantized fully connected (dense) layer.
// Input: [batch, inputSize], Weights: [outputSize, inputSize], Bias: [outputSize]
func opFullyConnected(interp *Interpreter, op model.Operator) error {
	if len(op.Inputs) < 3 || len(op.Outputs) != 1 {
		return fmt.Errorf("FULLY_CONNECTED: bad io")
	}
	sg := interp.model.Subgraphs[interp.sgIdx]

	inT := sg.Tensors[op.Inputs[0]]
	weightT := sg.Tensors[op.Inputs[1]]
	outT := sg.Tensors[op.Outputs[0]]

	inData := interp.tensors[op.Inputs[0]].data
	weightData := interp.tensors[op.Inputs[1]].data
	biasData := interp.tensors[op.Inputs[2]].data
	outData := interp.tensors[op.Outputs[0]].data

	activation := 0
	if v, ok := op.Options["fused_activation"]; ok {
		activation = v.(int)
	}

	// For FC, input is flattened. Total input size from weights.
	outputSize := weightT.Shape[0]
	inputSize := weightT.Shape[1]
	batch := model.TensorSize(inT.Shape) / inputSize

	inputZP := int32(0)
	if len(inT.Quant.ZeroPoint) > 0 {
		inputZP = int32(inT.Quant.ZeroPoint[0])
	}
	outputZP := int32(0)
	if len(outT.Quant.ZeroPoint) > 0 {
		outputZP = int32(outT.Quant.ZeroPoint[0])
	}

	inScale := float64(0)
	if len(inT.Quant.Scale) > 0 {
		inScale = float64(inT.Quant.Scale[0])
	}
	outScale := float64(0)
	if len(outT.Quant.Scale) > 0 {
		outScale = float64(outT.Quant.Scale[0])
	}
	// FC may have per-channel or single scale
	qmuls := make([]QuantMultiplier, outputSize)
	for c := 0; c < outputSize; c++ {
		wScale := float64(0)
		if c < len(weightT.Quant.Scale) {
			wScale = float64(weightT.Quant.Scale[c])
		} else if len(weightT.Quant.Scale) > 0 {
			wScale = float64(weightT.Quant.Scale[0])
		}
		if outScale == 0 {
			outScale = 1.0
		}
		qmuls[c] = QuantizeMultiplier(inScale * wScale / outScale)
	}

	actMin, actMax := activationBounds(activation, outputZP)

	for b := 0; b < batch; b++ {
		for oc := 0; oc < outputSize; oc++ {
			acc := int32(getBias32(biasData, oc))
			for ic := 0; ic < inputSize; ic++ {
				inVal := int32(int8(inData[b*inputSize+ic])) - inputZP
				wVal := int32(int8(weightData[oc*inputSize+ic]))
				acc += inVal * wVal
			}
			acc = MultiplyByQuantizedMultiplier(acc, qmuls[oc])
			acc += outputZP
			if acc < actMin {
				acc = actMin
			}
			if acc > actMax {
				acc = actMax
			}
			outData[b*outputSize+oc] = byte(int8(ClampInt8(acc)))
		}
	}
	return nil
}

// opLogistic applies the logistic (sigmoid) function.
func opLogistic(interp *Interpreter, op model.Operator) error {
	if len(op.Inputs) != 1 || len(op.Outputs) != 1 {
		return fmt.Errorf("LOGISTIC: bad io")
	}
	sg := interp.model.Subgraphs[interp.sgIdx]
	inT := sg.Tensors[op.Inputs[0]]
	outT := sg.Tensors[op.Outputs[0]]
	inData := interp.tensors[op.Inputs[0]].data
	outData := interp.tensors[op.Outputs[0]].data

	n := model.TensorSize(inT.Shape)

	inScale := float64(0)
	inZP := int32(0)
	if len(inT.Quant.Scale) > 0 {
		inScale = float64(inT.Quant.Scale[0])
	}
	if len(inT.Quant.ZeroPoint) > 0 {
		inZP = int32(inT.Quant.ZeroPoint[0])
	}
	outScale := float64(0)
	outZP := int32(0)
	if len(outT.Quant.Scale) > 0 {
		outScale = float64(outT.Quant.Scale[0])
	}
	if len(outT.Quant.ZeroPoint) > 0 {
		outZP = int32(outT.Quant.ZeroPoint[0])
	}

	for i := 0; i < n; i++ {
		// Dequantize input
		realVal := float64(int32(int8(inData[i]))-inZP) * inScale
		// Logistic
		sig := 1.0 / (1.0 + math.Exp(-realVal))
		// Quantize output
		outVal := int32(math.Round(sig/outScale)) + outZP
		if outT.Type == model.TensorUint8 {
			outData[i] = byte(ClampUint8(outVal))
		} else {
			outData[i] = byte(int8(ClampInt8(outVal)))
		}
	}
	return nil
}

// opQuantize re-quantizes a tensor from one scale/zp to another.
func opQuantize(interp *Interpreter, op model.Operator) error {
	if len(op.Inputs) != 1 || len(op.Outputs) != 1 {
		return fmt.Errorf("QUANTIZE: bad io")
	}
	sg := interp.model.Subgraphs[interp.sgIdx]
	inT := sg.Tensors[op.Inputs[0]]
	outT := sg.Tensors[op.Outputs[0]]
	inData := interp.tensors[op.Inputs[0]].data
	outData := interp.tensors[op.Outputs[0]].data

	n := model.TensorSize(inT.Shape)

	inScale := float64(0)
	inZP := int32(0)
	if len(inT.Quant.Scale) > 0 {
		inScale = float64(inT.Quant.Scale[0])
	}
	if len(inT.Quant.ZeroPoint) > 0 {
		inZP = int32(inT.Quant.ZeroPoint[0])
	}
	outScale := float64(0)
	outZP := int32(0)
	if len(outT.Quant.Scale) > 0 {
		outScale = float64(outT.Quant.Scale[0])
	}
	if len(outT.Quant.ZeroPoint) > 0 {
		outZP = int32(outT.Quant.ZeroPoint[0])
	}

	for i := 0; i < n; i++ {
		var realVal float64
		if inT.Type == model.TensorUint8 {
			realVal = float64(int32(inData[i])-inZP) * inScale
		} else {
			realVal = float64(int32(int8(inData[i]))-inZP) * inScale
		}
		outVal := int32(math.Round(realVal/outScale)) + outZP
		if outT.Type == model.TensorUint8 {
			outData[i] = byte(ClampUint8(outVal))
		} else {
			outData[i] = byte(int8(ClampInt8(outVal)))
		}
	}
	return nil
}

// --- Helper functions ---

func activationBounds(activation int, outputZP int32) (int32, int32) {
	actMin := int32(-128)
	actMax := int32(127)
	switch activation {
	case model.ActRelu:
		if outputZP > actMin {
			actMin = outputZP
		}
	case model.ActRelu6:
		if outputZP > actMin {
			actMin = outputZP
		}
		// For relu6, we'd need to compute quantized 6.0, but for MWW models
		// RELU is used, not RELU6.
	}
	return actMin, actMax
}

func getBias32(data []byte, idx int) int32 {
	off := idx * 4
	if off+4 > len(data) {
		return 0
	}
	return int32(data[off]) | int32(data[off+1])<<8 |
		int32(data[off+2])<<16 | int32(data[off+3])<<24
}

// opSplitV splits a tensor along an axis into multiple output tensors.
// Inputs: [value, size_splits, split_dim]
//   - value: the input tensor
//   - size_splits: 1-D int32 tensor with sizes for each split
//   - split_dim: scalar int32, the dimension to split along
//
// Outputs: num_splits tensors
func opSplitV(interp *Interpreter, op model.Operator) error {
	if len(op.Inputs) < 3 || len(op.Outputs) < 2 {
		return fmt.Errorf("SPLIT_V: need 3 inputs and >=2 outputs, got %d/%d",
			len(op.Inputs), len(op.Outputs))
	}

	sg := interp.model.Subgraphs[interp.sgIdx]
	inIdx := op.Inputs[0]
	splitSizesIdx := op.Inputs[1]
	axisIdx := op.Inputs[2]

	inT := sg.Tensors[inIdx]
	inData := interp.tensors[inIdx].data
	inShape := inT.Shape
	ndim := len(inShape)

	// Read axis (scalar int32)
	axisData := interp.tensors[axisIdx].data
	axis := getInt32FromBytes(axisData, 0)
	if axis < 0 {
		axis += ndim
	}
	if axis < 0 || axis >= ndim {
		return fmt.Errorf("SPLIT_V: axis %d out of range for %d dims", axis, ndim)
	}

	// Read size_splits (int32 vector)
	numSplits := len(op.Outputs)
	splitSizesData := interp.tensors[splitSizesIdx].data
	splitSizes := make([]int, numSplits)
	inferIdx := -1
	knownSum := 0
	for i := 0; i < numSplits; i++ {
		splitSizes[i] = getInt32FromBytes(splitSizesData, i)
		if splitSizes[i] == -1 {
			inferIdx = i
		} else {
			knownSum += splitSizes[i]
		}
	}
	if inferIdx >= 0 {
		splitSizes[inferIdx] = inShape[axis] - knownSum
	}

	// Compute strides for the input tensor
	// outerSize = product of dims before axis
	// axisSize = inShape[axis]
	// innerSize = product of dims after axis
	outerSize := 1
	for d := 0; d < axis; d++ {
		outerSize *= inShape[d]
	}
	innerSize := 1
	for d := axis + 1; d < ndim; d++ {
		innerSize *= inShape[d]
	}

	// bytesPerElement
	bpe := 1
	switch inT.Type {
	case model.TensorInt32:
		bpe = 4
	case model.TensorFloat32:
		bpe = 4
	case model.TensorInt16:
		bpe = 2
	}

	// Copy data into each output
	axisOffset := 0
	for s := 0; s < numSplits; s++ {
		outIdx := op.Outputs[s]
		out := &interp.tensors[outIdx]
		sliceSize := splitSizes[s]

		pos := 0
		for o := 0; o < outerSize; o++ {
			srcStart := (o*inShape[axis] + axisOffset) * innerSize * bpe
			chunkBytes := sliceSize * innerSize * bpe
			copy(out.data[pos:pos+chunkBytes], inData[srcStart:srcStart+chunkBytes])
			pos += chunkBytes
		}
		axisOffset += sliceSize
	}

	return nil
}
