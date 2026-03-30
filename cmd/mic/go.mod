module github.com/zserge/microwakeword/cmd/mic

go 1.22.0

require (
	github.com/gen2brain/malgo v0.11.24
	github.com/zserge/microwakeword v0.0.0
)

require github.com/google/flatbuffers v25.12.19+incompatible // indirect

replace github.com/zserge/microwakeword => ../..
