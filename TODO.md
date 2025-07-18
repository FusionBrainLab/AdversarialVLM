I have a python programm, which I will provide in the end. There is a functionality, I want you to add into my programm: 

1. Make an option to "restart optimizer". I mean that every "restart_num" of steps we will have smth like this:
dec_img = my_haha_backprocessing(x_0+x, device)
x_0 = my_haha_preprocessing(Image.fromarray(dec_img), device)
x = torch.zeros(x_0.shape).to(device)
x.requires_grad = True
optimizer = torch.optim.AdamW([x], lr=lr)

So the optimizer will be restarted, image reinicialized with learned attack added, and new attack will be started from zero.

2. Train only part of the image. For example, a mask can be used for x, to determine the location of trainable variables. 
Make some examples: trainable corner n*n, only k bottom lines. Another interesting option I need -- to train square n*n that will be randomly located in the image (so I can believe that attack will be robast to the movements).
Make it possible to choose using the args.

3. Suggest other ways of clamping that will not break gradients (make it possible to choose by args).

4. Make an option to start attack for a white image (without providing img_orig). 

5. Make sure different prompts are not alternated consecutively but are mixed within a single batch (i.e., there is no outer loop for question in questions and random questions are in inputs).

6*. Сделай возможность аттаки на несколько моделей одновременно: так чтобы разные модели висели на разных видеокартах, а атака училась одна универсальная для них всех. Можешь предложить и реализовать несколько подходов. 


TODO:

1. Random square реализовать так, чтобы один и тот же блок двигался по изображению (патч клеился к разным частям изображения)
2. Универсальность для разных изображений
3. Универсальность для разных моделей