name=$1
# python cornerReg.py --epochs=$2
mkdir models/$name
mv models/saved_modelPB_* models/$name
mkdir outs
for file in "models/$name"/*
do
    if [[ $file != *saved_modelPB_0 ]]
    then
        echo "Working with $file"
        # python loadSave.py $file
        # python runTFLITE.py lite_model.tflite |& tee  >(tail -1 >> outs/$name.txt)
        # mv lite_model.tflite models/$name/$file.tflite
    fi
done    
