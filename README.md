This is all the code for the anu comp8536 final report.
It includes three parts: vit,swin,convnext.
All files include the cutmix and baseline, to repeat the result you just need to set the random seed and set the lora rank.

In convnext part, there is no lora experiment because it is not a model based on transformer.
And its cifar experiment is basically sams as swin one, so what we do is when running the model we change the model name, other codes are the same 
