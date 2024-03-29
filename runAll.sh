name=$1
sudo mount /dev/xvdf /data
mkdir outs
rm outs/$name.txt
python main.py --epochs=$3 --datadir=$2 --saveName outs/$name.txt
# python main.py --epochs=$4 --datadir $2 --loadModel models/saved_model_35/weights.ckpt
mkdir models/$name
mv models/saved_model_* models/$name
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
        python runTFLITE.py --modelPath lite_model_pr.tflite --datadir $2 |& tee  >(tail -1 >> outs/$name.txt)
        echo "------------------------------------------------------" >> outs/$name.txt
        mv lite_model_pr.tflite $file.tflite
    fi
done    
mkdir models/$name/tfliteModels
mv models/$name/*.tflite models/$name/tfliteModels
