name=$1
python pruneNaive.py --epochs=$2 --datadir /data/card_synthetic_dataset_v2
mkdir models/$name
mv models/saved_modelPB_* models/$name
mkdir outs
rm outs/$name.txt
rm -rf models/$name/tfliteModels
for file in "models/$name"/*
do
    if [[ $file != *_0 ]] && [[ $file != *.tflite ]]
    then
        echo $file
        echo "Working with $file"
        python loadSave.py $file
        echo "Testing on $file" >> outs/$name.txt
        mv lite_model.tflite lite_model_pr.tflite
        python runTFLITE.py --modelPath lite_model_pr.tflite --datadir /data/card_synthetic_dataset_v2 |& tee  >(tail -1 >> outs/$name.txt)
        echo "------------------------------------------------------" >> outs/$name.txt
        mv lite_model_pr.tflite $file.tflite
    fi
done    
mkdir models/$name/tfliteModels
mv models/$name/*.tflite models/$name/tfliteModels
