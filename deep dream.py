# import numpy as np
# from functools import partial # helps with with creating new versions of functions with arguments filled in
# import PIL.Image #imaging library, helping us with modifying images
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import urllib.request # let us download data from the web
# import os # let us use operating system dependent functionality
# import zipfile

# def main():
#     #step 1: get google's pretrained neural network
#     url = 'http://storge.google.com/download.tensorflow.org/models/inception5h.zip'
#     data_dir = '.../data'
#     model_name = os.path.split(url)[-1]
#     local_zip_file = os.path.join(data_dir, model_name)
#     if not os.path.exists(local_zip_file):
#         #download
#         model_url = urllib.request.urlopen(url)
#         with open(local_zip_file, 'wb') as output:
#             output.write(model_url.read())  

#         #extract
#         with zipfile.ZipFile(local_zip_file,'r') as zip_ref:
#             zip_ref.extractall(data_dir) 

#     model_fn = 'tensorflow_inception_graph.pb'

#     #Step 2: creating tensorflow session and loading the model
#     graph = tf.Graph()
#     sess = tf.InteractiveSession(graph=graph)  
#     with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#     t_input = tf.placeholder(np.float32, name='input')  #define input tensor
#     imagenet_mean = 117.0
#     t_preproccesed = tf.expand_dims(t_input-imagenet_mean, 0)
#     tf.import_graph_def(graph_def, {'input':t_preproccesed}) 

#     layers = [ op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name ]
#     feature_nums = [int(graph.get_tensor_by_name(name+":0").get_shape()[-1]) for name in layers]

#     print("Number of layers",len(layers))
#     print("Total number of feature channels:",sum(feature_nums))


#     def strip_consts(graph_def, max_const_size=32):
#         """Strip large constant values from graph_def."""
#         strip_def = tf.GraphDef()
#         for n0 in graph_def.node:
#             n = strip_def.node.add() #pylint: disable=maybe-no-member
#             n.MergeFrom(n0)
#             if n.op == 'Const':
#                 tensor = n.attr['value'].tensor
#                 size = len(tensor.tensor_content)
#                 if size > max_const_size:
#                     tensor.tensor_content = "<stripped %d bytes>"%size
#         return strip_def
      
#     def rename_nodes(graph_def, rename_func):
#         res_def = tf.GraphDef()
#         for n0 in graph_def.node:
#             n = res_def.node.add() #pylint: disable=maybe-no-member
#             n.MergeFrom(n0)
#             n.name = rename_func(n.name)
#             for i, s in enumerate(n.input):
#                 n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
#         return res_def
      
#     def showarray(a):
#         a = np.uint8(np.clip(a, 0, 1)*255)
#         plt.imshow(a)
#         plt.show()
        
#     def visstd(a, s=0.1):
#         '''Normalize the image range for visualization'''
#         return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5
    
#     def T(layer):
#         '''Helper for getting layer output tensor'''
#         return graph.get_tensor_by_name("import/%s:0"%layer)
    
#     def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
#         t_score = tf.reduce_mean(t_obj) # defining the optimization objective
#         t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
        
#         img = img0.copy()
#         for _ in range(iter_n):
#             g, _ = sess.run([t_grad, t_score], {t_input:img})
#             # normalizing the gradient, so the same step size should work 
#             g /= g.std()+1e-8         # for different layers and networks
#             img += g*step
#         showarray(visstd(img))
        
#     def tffunc(*argtypes):
#         '''Helper that transforms TF-graph generating function into a regular one.
#         See "resize" function below.
#         '''
#         placeholders = list(map(tf.placeholder, argtypes))
#         def wrap(f):
#             out = f(*placeholders)
#             def wrapper(*args, **kw):
#                 return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
#             return wrapper
#         return wrap
    
#     def resize(img, size):
#         img = tf.expand_dims(img, 0)
#         return tf.image.resize_bilinear(img, size)[0,:,:,:]
#     resize = tffunc(np.float32, np.int32)(resize)
    
#     def calc_grad_tiled(img, t_grad, tile_size=512):
#         '''Compute the value of tensor t_grad over the image in a tiled way.
#         Random shifts are applied to the image to blur tile boundaries over 
#         multiple iterations.'''
#         sz = tile_size
#         h, w = img.shape[:2]
#         sx, sy = np.random.randint(sz, size=2)
#         img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
#         grad = np.zeros_like(img)
#         for y in range(0, max(h-sz//2, sz),sz):
#             for x in range(0, max(w-sz//2, sz),sz):
#                 sub = img_shift[y:y+sz,x:x+sz]
#                 g = sess.run(t_grad, {t_input:sub})
#                 grad[y:y+sz,x:x+sz] = g
#         return np.roll(np.roll(grad, -sx, 1), -sy, 0)



#     def render_deepdream(t_obj, img0= img_noise, iter_n = 10, step = 1.5, octave_n = 4, octave_scale = 1.4):
#         t_score = tf,reduce_mean(t_obj) #defining optimization objective
#         t_grad = tf.gradients(t_score, t_input)[0]

#         #splitting image into different octaves
#         img = img0.copy()
#         octaves = []
#         for _ in range(octave_n - 1):
#             hw = img.shape[:2]
#             lo = resize(img, np.int32(np.float32(hw)/octave_scale))
#             hi = img-resize(lo, hw)
#             img = lo
#             octaves.append(hi)
#         for octave in range(octave_n):
#             if octave > 0:
#                 hi = octaves[-octave]
#             img = resize(img, hi.shape[:2]) + hi
#             for _ in range(iter_n):
#                 g = calc_grad_tiled(img, t_grad)
#                 img += g*(step / (np.abs(g).mean()+1e-7))
            
#             showarray(img/255.0)




# #Step 3: pick a layer to enhance our image
# layer = 'mixed4d_3x3_botttleneck_pre_relu'
# channel = 139

# img0 = PIL.Image.open('Red_Kitten_01.jpg')
# img0 = np.float32(img0)

# #Step 4: Applying gradient ascent to the layer
# render_deepdream(tf.square(T('mixed4c')), img0) 

# if __name__ == '__main__':
#     main()


# import numpy as np
# import PIL.Image
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import os
# from pathlib import Path
# import requests
# from tqdm import tqdm
# import ssl

# # Add SSL context for macOS
# ssl._create_default_https_context = ssl._create_unverified_context

# class DeepDream:
#     def __init__(self, model_path=None):
#         """Initialize DeepDream with InceptionV3 model."""
#         try:
#             # Use InceptionV3 model that's built into TF instead of downloading separate file
#             base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
#         except Exception as e:
#             print("Error loading model:", e)
#             print("\nTrying alternative download method...")
#             # Alternative method to load model if direct download fails
#             base_model = tf.keras.models.load_model('inception_v3_no_top.h5') if os.path.exists('inception_v3_no_top.h5') else None
#             if base_model is None:
#                 raise Exception("Could not load model. Please ensure you have internet connection and valid SSL certificates.")
        
#         # Changed layers for stronger effect
#         names = ['mixed4', 'mixed5']  # Using deeper layers
#         layers = [base_model.get_layer(name).output for name in names]
#         self.model = tf.keras.Model(inputs=base_model.input, outputs=layers)
        
#     def preprocess(self, img):
#         """Prepare image for model input."""
#         # Convert to float32
#         img = tf.cast(img, tf.float32)
#         # Add batch dimension if not present
#         if len(img.shape) == 3:
#             img = tf.expand_dims(img, axis=0)
#         # Preprocess for InceptionV3
#         img = tf.keras.applications.inception_v3.preprocess_input(img)
#         return img
    
#     def deprocess(self, img):
#         """Convert processed image back to displayable format."""
#         # Remove the batch dimension
#         img = tf.squeeze(img)
#         img = 255 * (img + 1.0) / 2.0
#         return tf.cast(img, tf.uint8)
    
#     @tf.function
#     def calc_loss(self, img, layer_weights):
#         """Calculate loss for gradient ascent."""
#         outputs = self.model(img)
        
#         loss = tf.zeros(shape=())
        
#         for output, weight in zip(outputs, layer_weights):
#             loss += weight * tf.reduce_mean(tf.square(output))
            
#         return loss
    
#     def gradient_ascent_step(self, img, layer_weights, learning_rate):
#         """Perform one step of gradient ascent."""
#         with tf.GradientTape() as tape:
#             tape.watch(img)
#             loss = self.calc_loss(img, layer_weights)
            
#         gradients = tape.gradient(loss, img)
#         gradients = tf.math.l2_normalize(gradients)
#         img = img + gradients * learning_rate
#         return loss, img
    
#     def run_deep_dream(
#         self, 
#         img,
#         layer_weights=[2.0, 3.0],  # Increased weights for stronger effect
#         steps=150,  # More steps
#         step_size=0.05,  # Increased step size
#         octave_scale=1.4,  # Slightly increased scale
#         num_octaves=4
#     ):
#         """Run deep dream with octave processing."""
#         # Convert to tensor and add batch dimension
#         img = tf.constant(np.array(img))
#         original_shape = img.shape[:-1]
        
#         # Preprocess the image
#         img = self.preprocess(img)
        
#         # Create octave pyramid
#         octave_shapes = [original_shape]
#         for _ in range(num_octaves - 1):
#             shape = tf.cast(tf.cast(octave_shapes[-1], tf.float32) / octave_scale, tf.int32)
#             octave_shapes.append(tuple(shape.numpy()))
            
#         octave_shapes = octave_shapes[::-1]  # Work from smallest to largest
#         dream_img = tf.image.resize(img, octave_shapes[0])
        
#         for shape in tqdm(octave_shapes[1:], desc="Processing octaves"):
#             # Upscale dream image to new octave size
#             dream_img = tf.image.resize(dream_img, shape)
            
#             # Run gradient ascent
#             for step in range(steps):
#                 loss, dream_img = self.gradient_ascent_step(
#                     dream_img, 
#                     layer_weights,
#                     step_size
#                 )
                
#                 if step % 20 == 0:
#                     print(f'Loss at step {step}: {loss:.2f}')
                    
#         return self.deprocess(dream_img)
    
#     def save_result(self, img, output_path):
#         """Save the resulting image."""
#         PIL.Image.fromarray(img.numpy()).save(output_path)
        
# def main():
#     try:
#         # Create instance of DeepDream
#         print("Initializing DeepDream model...")
#         deep_dream = DeepDream()
        
#         # Load input image
#         input_path = "Red_Kitten_01.jpg"  # Change this to your input image path
#         if not os.path.exists(input_path):
#             raise FileNotFoundError(f"Input image not found at {input_path}")
            
#         print("Loading input image...")
#         img = PIL.Image.open(input_path)
#         img = np.array(img)
        
#         # Run deep dream
#         print("Running DeepDream process...")
#         result = deep_dream.run_deep_dream(
#             img,
#             layer_weights=[2.0, 3.0],  # Stronger weights
#             steps=150,  # More steps
#             step_size=0.05,  # Larger step size
#             octave_scale=1.4,
#             num_octaves=4
#         )
        
#         # Save original and result side by side for comparison
#         plt.figure(figsize=(20, 10))
        
#         plt.subplot(1, 2, 1)
#         plt.title('Original Image')
#         plt.imshow(img)
#         plt.axis('off')
        
#         plt.subplot(1, 2, 2)
#         plt.title('DeepDream Result')
#         plt.imshow(result.numpy())
#         plt.axis('off')
        
#         plt.tight_layout()
#         plt.show()
        
#         # Save result
#         output_path = "deep_dream_result.jpg"
#         deep_dream.save_result(result, output_path)
#         print(f"Result saved to {output_path}")
        
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         print("\nTroubleshooting steps:")
#         print("1. Check that your input image is a valid image file")
#         print("2. Ensure you have sufficient memory for processing")
#         print("3. Try reducing the image size if it's too large")
#         print("4. Make sure you have write permissions in the output directory")

# if __name__ == '__main__':
#     main()




# ----------------------------------------------------------------------------------------------------------------








# import numpy as np
# from functools import partial
# import PIL.Image
# import tensorflow as tf
# tf.compat.v1.enable_eager_execution()  # Enables immediate tensor evaluation
# import matplotlib.pyplot as plt
# import urllib.request
# import os
# import zipfile

# def main():
#     # Step 1: Download the pre-trained Inception model
#     url = 'http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
#     data_dir = './data'
#     model_name = os.path.split(url)[-1]
#     local_zip_file = os.path.join(data_dir, model_name)

#     if not os.path.exists(local_zip_file):
#         os.makedirs(data_dir, exist_ok=True)
#         model_url = urllib.request.urlopen(url)
#         with open(local_zip_file, 'wb') as output:
#             output.write(model_url.read())

#         with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
#             zip_ref.extractall(data_dir)

#     model_fn = 'tensorflow_inception_graph.pb'

#     # Step 2: Creating TensorFlow session and loading the model
#     tf.compat.v1.disable_eager_execution()
#     graph = tf.Graph()
#     sess = tf.compat.v1.InteractiveSession(graph=graph)

#     with tf.io.gfile.GFile(os.path.join(data_dir, model_fn), 'rb') as f:
#         graph_def = tf.compat.v1.GraphDef()
#         graph_def.ParseFromString(f.read())

#     # Placeholder for image input
#     t_input = tf.compat.v1.placeholder(np.float32, name='input')
#     imagenet_mean = 117.0
#     t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)

#     # Import the graph and link it with the input
#     tf.import_graph_def(graph_def, {'input': t_preprocessed})

#     # Extract convolutional layers and features
#     layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
#     feature_nums = [int(graph.get_tensor_by_name(name + ":0").get_shape()[-1]) for name in layers]

#     print(f"Number of layers: {len(layers)}")
#     print(f"Total number of feature channels: {sum(feature_nums)}")

#     # Utility functions
#     def showarray(a):
#         try:
#             a = a.numpy()  # Try converting to NumPy if using TF2.x
#         except AttributeError:
#             a = sess.run(a)  # Fall back to TF1.x session run
#         a = np.uint8(np.clip(a, 0, 255))
#         plt.imshow(a)
#         plt.axis('off')
#         plt.show()



#     def visstd(a, s=0.1):
#         """Normalize the image range for visualization."""
#         return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5

#     def T(layer):
#         """Helper for getting layer output tensor."""
#         return graph.get_tensor_by_name(f"import/{layer}:0")

#     # Gradient ascent visualization for basic deep dream effect
#     def render_naive(t_obj, img0, iter_n=20, step=1.0):
#         """Apply naive gradient ascent to visualize the features."""
#         t_score = tf.reduce_mean(t_obj)  # Define the optimization objective
#         t_grad = tf.gradients(t_score, t_input)[0]  # Compute gradients automatically

#         img = img0.copy()
#         for _ in range(iter_n):
#             g, score = sess.run([t_grad, t_score], {t_input: img})
#             g /= (g.std() + 1e-8)  # Normalize the gradient
#             img += g * step  # Apply gradient step
#         showarray(visstd(img) * 255)

#     def tffunc(*argtypes):
#         """Transforms a TensorFlow graph-generating function into a callable Python function."""
#         def wrap(f):
#             @tf.function  # Use TensorFlow's graph decorator
#             def wrapper(*args):
#                 return f(*args)
#             return wrapper
#         return wrap

#     # Resize function using TensorFlow for image manipulation
#     def resize(img, size):
#         img = tf.ensure_shape(img, [None, None, 3])  # Define shape explicitly
#         img = tf.expand_dims(img, 0)
#         resized = tf.image.resize(img, size)[0, :, :, :]
#         return resized


#     resize = tffunc(np.float32, np.int32)(resize)

#     # Tiled gradient computation for deeper dreams
#     def calc_grad_tiled(img, t_grad, tile_size=512):
#         """Compute the gradient in a tiled way, ensuring proper tensor evaluation."""
#         sz = tile_size
#         h, w = img.shape[:2]
#         sx, sy = np.random.randint(sz, size=2)

#         # Explicitly evaluate the tensor to avoid symbolic errors
#         img_np = sess.run(img) if isinstance(img, tf.Tensor) else img

#         img_shift = np.roll(np.roll(img_np, sx, axis=1), sy, axis=0)
#         grad = np.zeros_like(img_shift)

#         for y in range(0, max(h - sz // 2, sz), sz):
#             for x in range(0, max(w - sz // 2, sz), sz):
#                 sub = img_shift[y:y + sz, x:x + sz]
#                 # Explicitly evaluate the sub-region if it's still a tensor
#                 sub_np = sess.run(sub) if isinstance(sub, tf.Tensor) else sub
#                 g = sess.run(t_grad, {t_input: sub_np})
#                 grad[y:y + sz, x:x + sz] = g

#         # Return NumPy array since TensorFlow tensors cannot be mixed directly
#         return np.roll(np.roll(grad, -sx, axis=1), -sy, axis=0)




#     # Advanced deepdream function with octaves for enhanced visuals
#     def render_deepdream(t_obj, img0, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
#         """Apply deep dream with multiple scales."""
#         t_score = tf.reduce_mean(t_obj)
#         t_grad = tf.gradients(t_score, t_input)[0]

#         img = img0.copy()
#         octaves = []
#         for _ in range(octave_n - 1):
#             hw = img.shape[:2]
#             lo = resize(img, np.int32(np.float32(hw) / octave_scale))
#             hi = img - resize(lo, hw)
#             img = lo
#             octaves.append(hi)

#         for octave in range(octave_n):
#             if octave > 0:
#                 hi = octaves[-octave]
#                 img = resize(img, hi.shape[:2]) + hi
#             for _ in range(iter_n):
#                 g = calc_grad_tiled(img, t_grad)
#                 img += g * (step / (np.abs(g).mean() + 1e-7))
#             showarray(img / 255.0)

#     # Step 3: Load an image and apply deep dream
#     img0 = PIL.Image.open('/Users/gururaj/Documents/project=deep dream/Red_Kitten_01.jpg')
#     img0 = np.float32(img0)
#     layer = 'mixed4d_3x3_bottleneck_pre_relu'
#     channel = 139

#     print("Running DeepDream...")
#     render_deepdream(T(layer)[:, :, :, channel], img0)

# if __name__ == "__main__":
#     main()




#----------------------------------------------------------------------------------------------------------------



import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # Use TensorFlow 1.x behavior
import matplotlib.pyplot as plt
import urllib.request
import os
import zipfile

def main():
    # Step 1: Download the pre-trained Inception model
    url = 'http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data_dir = './data'
    model_name = os.path.split(url)[-1]
    local_zip_file = os.path.join(data_dir, model_name)

    if not os.path.exists(local_zip_file):
        os.makedirs(data_dir, exist_ok=True)
        model_url = urllib.request.urlopen(url)
        with open(local_zip_file, 'wb') as output:
            output.write(model_url.read())

        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    model_fn = 'tensorflow_inception_graph.pb'

    # Step 2: Creating TensorFlow session and loading the model
    graph = tf.Graph()
    sess = tf.compat.v1.InteractiveSession(graph=graph)

    with tf.io.gfile.GFile(os.path.join(data_dir, model_fn), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Placeholder for image input
    t_input = tf.compat.v1.placeholder(np.float32, name='input')
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)

    # Import the graph and link it with the input
    tf.import_graph_def(graph_def, {'input': t_preprocessed})

    # Extract convolutional layers and features
    layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name + ":0").get_shape()[-1]) for name in layers]

    print(f"Number of layers: {len(layers)}")
    print(f"Total number of feature channels: {sum(feature_nums)}")

    # Utility functions
    def showarray(a):
        a = np.uint8(np.clip(a, 0, 255))  # Clip values to [0, 255]
        plt.imshow(a)
        plt.axis('off')
        plt.show()

    def visstd(a, s=0.1):
        """Normalize the image range for visualization."""
        return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5

    def T(layer):
        """Helper for getting layer output tensor."""
        return graph.get_tensor_by_name(f"import/{layer}:0")

    # Gradient ascent visualization for basic deep dream effect
    def render_naive(t_obj, img0, iter_n=20, step=1.0):
        """Apply naive gradient ascent to visualize the features."""
        t_score = tf.reduce_mean(t_obj)  # Define the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0]  # Compute gradients automatically

        img = img0.copy()
        for _ in range(iter_n):
            g, score = sess.run([t_grad, t_score], {t_input: img})
            g /= (g.std() + 1e-8)  # Normalize the gradient
            img += g * step  # Apply gradient step
        showarray(visstd(img) * 255)

    # Resize function using TensorFlow for image manipulation
    def resize(img, size):
        img = tf.image.resize(img, size)  # Resize operation
        return sess.run(img)  # Evaluate tensor and return as NumPy array

    # Tiled gradient computation for deeper dreams
    def calc_grad_tiled(img, t_grad, tile_size=512):
        """Compute the gradient in a tiled way."""
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)

        img_shift = np.roll(np.roll(img, sx, axis=1), sy, axis=0)
        grad = np.zeros_like(img_shift)

        for y in range(0, max(h - sz // 2, sz), sz):
            for x in range(0, max(w - sz // 2, sz), sz):
                sub = img_shift[y:y + sz, x:x + sz]
                g = sess.run(t_grad, {t_input: sub})
                grad[y:y + sz, x:x + sz] = g

        return np.roll(np.roll(grad, -sx, axis=1), -sy, axis=0)

    # Advanced deepdream function with octaves for enhanced visuals
    def render_deepdream(t_obj, img0, iter_n=10, step=50, octave_n=4, octave_scale=2.1):
        """Apply deep dream with multiple scales."""
        t_score = tf.reduce_mean(t_obj)
        t_grad = tf.gradients(t_score, t_input)[0]

        img = img0.copy()
        octaves = []
        for _ in range(octave_n - 1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw) / octave_scale))
            hi = img - resize(lo, hw)
            img = lo
            octaves.append(hi)

        for octave in range(octave_n):
            if octave > 0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2]) + hi
            for _ in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                img += g * (step / (np.abs(g).mean() + 1e-7))
            showarray(visstd(img) * 255)

    # Step 3: Load an image and apply deep dream
    img0 = PIL.Image.open('Red_Kitten_01.jpg')
    img0 = np.float32(img0)
    layer = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139

    print("Running DeepDream...")
    render_deepdream(T(layer)[:, :, :, channel], img0)

if __name__ == "__main__":
    main()

