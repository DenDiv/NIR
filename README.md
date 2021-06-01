# Masked faces recognition system<br/>
## Currently work on detection masked faces<br/>
### Masked faces detection<br/>
#### Test<br/>
Firstly you need to create predictions on MAFA and WIDER (optional), there is an example of MTCNN below<br/>
>MTCNN example:<br/>
>to create predictions on WIDER VAL:<br/>
>python mtcnn_pred.py -d data/WIDER/WIDER_val -m WIDER --save_folder predictions/WIDER_preds/mtcnn_preds<br/>
>to create predictions on MAFA (masked faces dataset):<br/>
>python mtcnn_pred.py -d data/MAFA/test-images -m MAFA --save_folder predictions/MAFA_preds/mtcnn_preds<br/>

Now you can plot PR curve for MAFA and WIDER<br/>
Create plots .m files:<br/>
> for WIDER evaluation:
> in Octave CLI:
> cd /path_to/.../WIDER_eval_tools
> wider_eval("../predictions/WIDER_preds/mtcnn_preds", "mtcnn_our")
> python plot_wider
> for MAFA evaluation:
> python converters/mafa2pascal --mat_file data/MAFA/MAFA-Label-Test/LabelTestAll.mat --save_path data/MAFA/test-images --del_invalid True
> python converters/pascal2metric --gt_dir data/MAFA/test-images/annotations --save_dir data/MAFA/test-images/eval_annotations
> python AP_eval --gt data/MAFA/test-images/eval_annotations --predicted predictions/MAFA_preds/mtcnn_preds --name mtcnn