# Interact with Drava

<img src='../assets/interface.png' width='850px' >

### Concept View

In the *Concept View*, all latent dimensions will be ranked based on their saliency scoreas from high to low.  
You can **remove a dimension** by clicking on the cross icon, **rename a dimension** by directly editing upon it, or **add a dimension** using the drop down menu in the right top corner. 


<details>
  <summary>Click to View a Demo :eyes: </summary>
  <div> 

[context_demo](../assets/context_demo.mp4 ':include :type=iframe width=600px height=300px')
  
  </div>
</details>

For each dimension, the histogram bars are clickable, performing as a toggle button to **filter items**.  
Items in the *Spatial View* and the *Item Browser* will be updated accordingly. 

<details>
  <summary>Click to View a Demo :eyes: </summary>
  <div> 

[context_demo](../assets/filter_demo.mp4 ':include :type=iframe width=600px height=500px')
  
  </div>
</details>

### Spatial View

The *Spatial View* is an optional view for items that have spatial information.
All items will be arranged based on their spatial relationships.  
Users can **zoom** and **pan** to obtain an overview or inspect details.

<details>
  <summary>Click to View a Demo :eyes: </summary>
  <div> 

[context_demo](../assets/spatial_demo.mp4 ':include :type=iframe width=600px height=350px')
  
  </div>
</details>

### Item Browser

The *Item Browser* provide a *Config Panel* that allows users to easily change the **Arrange**, **Groupping**, **Sizes**, **Labels**, and **Summary** of items.

<details>
  <summary>Click to View a Demo :eyes: </summary>
  <div> 

[context_demo](../assets/config_demo.mp4 ':include :type=iframe width=600px height=500px')
  
  </div>
</details>

**In-Place Browse:**  
Click on an item group. Mouse over an item preview to browse it in place.

<details>
  <summary>Click to View a Demo :eyes: </summary>
  <div> 

[context_demo](../assets/inplace_demo.mp4 ':include :type=iframe width=600px height=500px')
  
  </div>
</details>

**Freeform Lasso:**  
Hold down `Shift`, click and hold on to draw the lasso.  
All items and item groups that are selected by the lasso will be merged into one group. 

**Split a Group:**   
First, right click on the group to show the *context menu* and select "Browse Separately".  
Then, in the new layer, use a freeform lasso to select the subgroup for splitting.  
Finally, right click on the subgroup and select "Extract This Pile" in the Context Menu.

<details>
  <summary>Click to View a Demo :eyes: </summary>
  <div> 

[context_demo](../assets/split_demo.mp4 ':include :type=iframe width=600px height=500px')
  
  </div>
</details>

Meanwhile, the *Config Panel* provides a button to split all groups throuch one click.

**Merge Groups:**  
Groups can be merged through 1) drag and drop; or 2) a freeform lasso selection.

**Extract an Item from Group:**  
You can extract an item from its group via either of the following two methods.  
a) through shortcut:  
First, activate the in-place browse for the item you want to extract.
Then, hold down `Option` (`Alt` for windowns) and click the item to extract.  
b) through context menu:  
First, activate the in-place browse for the item you want to extract.
Then, right click to show the *context menu* and select "Extract This Item".

<details>
  <summary>Click to View a Demo :eyes: </summary>
  <div> 

[context_demo](../assets/extract_demo.mp4 ':include :type=iframe width=600px height=500px')
  
  </div>
</details>

## Update Concept

Drava supports two concept updating mechanisms: local and global.
In local update, Drava will remember the user refinement but do not update the underlying model. On the contrary, global update fine-tunes the
concept adaptor and update the values for all other items at this dimension accordingly. 
The global refinement is triggered by clicking the "update concept" button. Since it can be hard for users to assign an exact numerical
value to describe a certain concept, global refinement is only applied when items are grouped into several classes for a
certain concept (i.e., items are grouped by X and x-axis indicates one latent dimension value).