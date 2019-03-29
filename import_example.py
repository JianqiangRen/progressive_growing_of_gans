import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import datetime
import random
# Initialize TensorFlow session.
tf.InteractiveSession()

# Import official CelebA-HQ networks.
# with open('karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
with open('network-snapshot-008800.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

for epoch in range(100000):
    seed = random.randint(0,10000)
    # Generate latent vectors.
    latents = np.random.RandomState(seed).randn(10000, *Gs.input_shapes[0][1:]) # 1000 random latents
    
    for k in range(100):
        idx = []
        for i in range(10):
            idx.append(random.randint(0,10000-1))
        
        latents = latents[idx] # hand-picked top-10
        
        # latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10
        
        print("latents shape:", np.shape(latents))
        # Generate dummy labels (not used by the official networks).
        labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])
        
        # Run the generator to produce a set of images.
        images = Gs.run(latents, labels)
        
        # Convert images to PIL-compatible format.
        images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
        images = images.transpose(0, 2, 3, 1) # NCHW => NHWC
        
        dt = datetime.datetime.now().strftime("%Y-%m-%d%H%M%S")
        
        # Save images as PNG.
        for idx in range(images.shape[0]):
            PIL.Image.fromarray(images[idx], 'RGB').save('output/' +dt +'img%d.png' % idx)
