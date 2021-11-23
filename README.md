# About this B-Spline Fitting algorithm

This algorithm aims at describing a given curve from hand sketches using a B-Spline as precisely as possible while not having any prior information (especially the number of control points) about the curve. So far this algorithm is still in progress.

The algorithm consists of two parts:

- **\[Completed\]** Given a fixed number of control points, solve for the optimization problem so that the reconstruction error is minimized by placing the control points.
    - Use the method **splprep()** in **scipy** package for stability.
    - While the package itself did not provide a handle controlling the number of control points, designing the knot vector based on curve types can produce desired number of control points.
    - Compatible for both open and closed curves.
- **\[In progress\]** For an arbitrary curve after segmentation of the sketch interface, find the optimal number of control points.
    - Generate a dataset by randomly sampling B-Spline curves created with different number of  control points. 
    - Construct a network learns the optimal number of control points and use it for prediction.
    - Use the trained network for prediction.

So far the workflow for the second part has been successfully created and initialized. The completed works are as follows.

- Compute knot vectors for both open and closed curves.
- Sample control points of different numbers.
- Generate curve images and save all the related information as a dictionary object
- Build a complex model similar to ResNet-18 and a relatively simple  and straight forward model to compare the performances
- Construct the training and validation process and check the models' performance when trained with different types of curves (e.g. open curves only, closed curves only, and both open and closed curves)
- Use the trained model to predict the number of control points given a numpy.ndarray representing the sampled curve from the sketch interface.

Further improvements are required since the current model is not working well on the validation dataset. Possible future works are as follows.

  - Improve the generated dataset
      - Make curves controlled by different number of control points more distinct
      - Make the distribution of sampled curves more rational and similar to real drawings
  - Train the complex model with a dataset bigger in size to avoid overfitting.
  - Modify the model structure and hyperparameters for better performance.

The figures showing the performance of current models are in the **Figure** folder.

- Two folders **complex** and **simple** indicating which model is evaluted
- In **complex** and **simple**, three folders **open**, **closed** and **all** indicating which type of curves are fed into the model in training process.
- The confusion matrix on validation dataset  after each epoch is listed.
- The plots training loss, validation loss and validation accuracy are included.

# How to use

- The fitting methods (find optimal control points given fixed number of control points) can be viewed in **fitting.py**. The function needs to be imported from this script.
- All the scripts do not accept external arguments. Instead, all the settings are in **config.py**. 
  - Edit *N_min* and *N_max* to modify the range of number of control points
  - Edit *Images_op* and *Images_cl* to change the number of created open (op) and closed (cl) curves for training and validation. Expected input is *[num_training, num_validation]*.
  - Edit *Deg* to modify the degree of created curves.
  - Edit *Width, Height, Dpi, Linewidth* to modify the created images.
  - Edit *Epoch* to modify the number of epochs for one training process.
  - Edit *Model_types* to assign the list of models to be trained.
  - Edit *Curve_types* to assign the list of curve types for training the  model.
- The codes related to generating the spline dataset can be viewed in **datagenerator.py**. 
  - Run this script after modify the settings in **config.py** to generate the desired spline dataset.
  - Corresponding dataset object is defined in **dataset.py**.
- The complex model and the simple model are defined in **model.py**.
- The evaluation method is defined in **evaluate.py**.
- The training and validation process is coded in **train&validate.py**.
  - Run this script to train the assigned model with created dataset.
- The prediction method is defined in **inference.py**.
  - Due to bad performance of the current models, I did not test this method and check its compatibility with the fitting methods.