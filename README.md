# medicalViT_MAE


![스크린샷 2025-02-06 오후 4 43 11](https://github.com/user-attachments/assets/b791faa4-170f-467d-87a4-901df605e63e)


Model overview. MAE consists of an encoder and a decoder, both of which are built using transformer encoder blocks. Model's encoder takes an input image where some parts are masked. It receives only the unmasked patches as input and is trained to understand the overall structure of the image, producing embedding vectors as output. 
Model’s decoder uses these embedding vectors to reconstruct the original image. It reconstructs all the pixels of the original image, including the masked areas.

