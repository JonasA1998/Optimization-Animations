# Visualization_Project
Contains functions to animate/plot *GradientDescend* vs. *Newtons Method* for different Objective Functions such as the Rosenbrockfunction (Often used to compare convergence speed/behavior). 

You can define different functions $f:\mathbb{R^{2}}\to\mathbb{R}$ or $f:\mathbb{R}\to\mathbb{R}$ and 1 or many starting points $x_{0}\in\mathbb{R}^{n}$ n $\in$ {0,1} and the following code with animate the convergence of the 2 optimization methods 


```python
rosen = lambda x : (1-x[0])**2 + 105.*(x[1]-x[0]**2)**2
x0,x_vec,max_len=opti.grad_des(rosen,[[0,0]],.5,100,True)
x0,x_ar,max_len1 = opti.ungedNewt(rosen,[[0,0]],.5,100,True)
(...) # in notebook
# Creating the Animation object
anim = animation.FuncAnimation(
    fig, update_lines, max_len, fargs=(x_vec, lines), interval=500)

```

![Animation](https://user-images.githubusercontent.com/95909459/224712635-864c55b3-aed5-4c3d-885e-6ead45174a49.gif)

<sup>**Note:(Its possible to use armijo stepsize regularization if needed with the argument stepreg=True, default stepsize = 1)** 
    
Also you can compare the convergence speed with this widget plot, for different starting points:
![widget](https://user-images.githubusercontent.com/95909459/224715701-4796793e-cc96-412e-81e4-61bf035af921.png)
    
Other Plots:
![lines](https://user-images.githubusercontent.com/95909459/224714155-55164753-c5fc-4fe1-896f-4f301e9c1568.gif)
    
![newton](https://user-images.githubusercontent.com/95909459/224714494-b570e4ee-2fe3-4199-99fd-ffd0ff6d9601.png)

