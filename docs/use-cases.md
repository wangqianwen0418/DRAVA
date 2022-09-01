# Use Cases

?> This page illustrates the application scenarios of Drava.

<table>
  <tr>
    <td> 
        <a href='/#/use-cases?id=simple-shapes'>
            <div style="background-image: url(https://user-images.githubusercontent.com/9922882/187742539-cc664158-c581-411a-8257-cc61afc4768a.png); width:200px; height:100px; background-size: cover;"  ></div>
            <br/>
            <span>Simple Shapes (dsprites dataset)</span>
        </a>
    </td>
    <td>
        <a href='/#/use-cases?id=celebrity-images'>
            <div style="background-image: url(https://user-images.githubusercontent.com/9922882/187742536-402af1a2-04ec-42a3-b682-ebb4c406fee5.png); width:200px; height:100px; background-size: cover;"  ></div>
            <br/>
            <span>Celebrity Images (celeba dataset)</span>
        </a>
    </td>
     <td>
        <a href='/#/use-cases?id=genomic-interaction-matrix'>
            <div style="background-image: url(https://user-images.githubusercontent.com/9922882/187742534-5e1bc225-b6b8-421f-82d7-4942dc130e90.png); width:200px; height:100px; background-size: cover;"  ></div>
            <br/>
            <span>Genomic Interaction Matrices</span>
        </a>
    </td>
   </tr> 
   <tr>
    <td>
        <a href='/#/use-cases?id=breast-cancer-specimen'>
            <div style="background-image: url(https://user-images.githubusercontent.com/9922882/187742532-e6034506-0ca0-4233-af0e-0f52f02ae18c.png); width:200px; height:100px; background-size: cover;"  ></div>
            <br/>
            <span>Breast Cancer Specimen</span>
        </a>
    </td>
    <td>
        <a href='/#/use-cases?id=celebrity-images'>
            <div style="background-image: url(https://user-images.githubusercontent.com/9922882/187742534-5e1bc225-b6b8-421f-82d7-4942dc130e90.png); width:200px; height:100px; background-size: cover;"  ></div>
            <br/>
            <span>Single Cell Masks</span>
        </a>
    </td>
    <td>
        <a href = '/#/your-data'>
              <div style="width:200px; height:100px; background-size: cover;" >
              </div>
            <span>Try Drava on your dataset</span>
        </a>
    </td>
  </tr>
</table>

### Simple Shapes
<center><img src="https://user-images.githubusercontent.com/9922882/187742539-cc664158-c581-411a-8257-cc61afc4768a.png" style='max-width: 600px'/></center>

*Figure 1. (a) Images are arranged based on a UMAP projection, which put images together even though the positions of shapes in these images are different. (b) Images are arranged based on the shape position. (c) Images are arranged based on a visual concept that is related to the scales of shape, but the left-most side are mostly squares.*

#### Data and Model
This scenario uses the [dsprites image dataset](https://github.com/deepmind/dsprites-dataset/), which consists of three types of simple shapes (i.e., square, ellipse, heart) with different scales, positions, and orientations. We uniformly sampled 1,000 items. The DRL model has four convolution blocks, each of which has 32 channels, a kernel of size 4, and a stride of 2. The latent vector has 8 dimensions. Even though this is a simple dataset, this can work as a proxy of more complicated datasets, such as the bounding boxes in object detection or the masks for cell segmentation in tissue imaging.

#### Arranging Items based on Concepts of Interest
To start with, we display all items in a 2D space using a UMAP projection, a method that is commonly used for visualizing items with latent vectors. While the UMAP successfully put items with similar shapes and scales close to one another, the shape position information is mostly ignored, as shown in Figure 1a. The position information can be important for some analysis tasks, e.g., object detection in autopilot. 

Based on the synthesized images in the Concept View, the position-related information has been successfully extracted in two dimensions, which we rename to `dim_x` and `dim_y`<!--(Figure 1E)-->. As shown in Figure 1b, all images are arranged and grouped based on the x and y position of the shape in the image. We choose see-through technique to summarize a group, which enables us to inspect the overall positions of the shapes without browsing individual items one by one. 

#### Refine a Semantic Dimension
The scale, i.e.size, of the shapes is also a vital piece of information for some analyses and has been successfully extracted in a latent dimension (named as `dim_size`). We further verify this semantic dimension in the Item Browser, setting `dim_size` as 𝑥 axis and its deviation 𝜎 as the 𝑦 axis. While all items are sorted based on their size from left to right, we find that items on the left side are all squares (Figure 1c). This might be caused by the fact that ellipse and heart, even with the same scale, are smaller than square in terms of absolute pixel area.

To obtain a semantic dimension that better matches the analysis purpose and indicates the scale regardless of shape types, we refine `dim_size` using the concept adaptor. We first group the current item groups into three main groups, indicating large, medium, and small scales, respectively. After clicking the update concept button, the concept adaptor is initialized based on our grouping. To refine the grouping, we use the browse separately function to examine each group and update the group mainly by moving items of ellipse or heart shape from the medium group to the large group. After several updates, we click the update concept button again. The concept adaptor is fine-tuned based on the refined item groups and updates the grouping of all items. After several iterations, we obtain three groups that more accurately reflect the scale of shapes without the influence of shape types.

### Celebrity Images
<center><img src="https://user-images.githubusercontent.com/9922882/187742536-402af1a2-04ec-42a3-b682-ebb4c406fee5.png" style='max-width: 600px'/></center>

*Figure 2. (a) Items are grouped based on a visual concept that is related to the skin color. (b) Items in (a) are arranged by adding another visual concept that is related to the background darkness as the 𝑦 axis.*

#### Data and Model
This usage scenario uses the celebrity images from the [CelebA dataset](https://openaccess.thecvf.com/content_iccv_2015/html/Liu_Deep_Learning_Face_ICCV_2015_paper.html). The DRL model is trained on the complete dataset and we randomly sample 1,000 items for the exploration in Drava. The DRL model has five convolution blocks, each of which contains a kernel of size 3, a stride of 2, and 32, 64, 128, 256, 512 channels, respectively. The latent vector has 20 dimensions.

#### Investigate Dataset Diversity
Collecting a diverse dataset is important in machine learning to improve the model performance in real-world deployment and avoid algorithmic discrimination of certain populations ([:link:](https://ieeexplore.ieee.org/abstract/document/9222272)). The concepts extracted by Drava offer an effective approach to investigate the diversity of a dataset.

Based on the synthesized images in the Concept View, we can affirm that diverse visual concepts exist in the analyzed data items. The analyzed items vary in a number of aspects, including the emotional expression, gender, angle, skin color, background color, hair length, and hair styles. To further verify our interpretation of the semantics of individual dimensions, we can interactively change the latent vector to update the synthesized images and group items based on their latent values at a selected dimension<!--(Figure 1D)-->. 

#### Investigate Dataset Balance
We then analyze the item distribution along individual concepts as dataset imbalance can introduce bias during model training and impair model performance. For example, for the "skin color" concept, a dataset with a large number of items with fair skin and only a small number of items with dark skin can lead to a machine learning model that has poor performance on the latter. As shown in Figure 2a, we arrange and group items based on `dim_9`, which captures skin color based on the synthesized images. Through browsing items in these groups, we find only the right several groups include people with dark skin (Figure 2A1), indicating a relatively small portion. When we browse individual items in each group (Figure 2A2), we can find that this portion is even smaller since the model considers people with dark skin and people with shadows on their faces as similar. This observation implies a data imbalance related to skin color, which can introduce a bias into a model trained on it. 

#### Confirm Concept Association
Based on Figure 2a, we suspect that there is a correlation between dark skin and dark background. Such correlations can be treated as causalities by machine learning models ([:link:](https://ieeexplore.ieee.org/abstract/document/9552218)) and needs to be avoided. We confirm this suspicion by arranging all items using `dim_9` (skin tone) as the 𝑥-axis and `dim_16` (background darkness) as the 𝑦-axis. The resulting distribution (Figure 2b) dispels our suspicion. Even though the distribution is not uniform, the dataset contains both items that have fair skin and dark background (B1) and items that have dark skin and light background (B2).

### Genomic Interaction Matrix
<center><img src="https://user-images.githubusercontent.com/9922882/187742534-5e1bc225-b6b8-421f-82d7-4942dc130e90.png" style='max-width: 600px'/></center>

*Figure 3.  (A1-4) Four typical items that vary on three concepts: the thickness of the diagonal, presence of nested squares, asymmetric structure of the nested squares. (B) A group mixes the items with nested squares (A2, A3) with items with thick diagonal (A4, orange box). (C) Arranging items using dim_6 as 𝑥 axis and dim_th as 𝑦 axis.*

#### Data and Model
This usage scenario uses a genome interaction matrix for the HFFc6 cell line published by [Rao et al.](https://www.cell.com/fulltext/S0092-8674(14)01497-4). The matrices describe the interactions between different genomic locations, which is related to the folding of DNA and affects regulation of gene expression. In a genome interaction matrix, rows and columns represent genomic locations and the color intensity indicates the interaction probability between a pair of locations. When visually exploring a large matrix, e.g.for 3M × 3M for human genome, experts typically examine regions of interest (ROI) that have unique visual patterns and indicate specific biological events. We generate small multiples for one specific type of ROIs called Topologically Associated Domains (TAD), which are visually represented as squares that are presumably organized hierarchically. We first extract TADs from the interaction matrix using [OnTAD](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1893-y) and then use the DRL model to generate a latent vector for each TAD. We demonstrate Drava using the 855 TADs extracted by OnTAD from chromosome 5 of the HFFc6 cell line. The DRL model has three convolution blocks with filter sizes of 7, 5, 3 and channel sizes of 32, 64, 128, respectively. The latent vector has 8 dimensions.

#### Understand Data through Concepts
The visual appearance of TADs in a heatmap can serve as effective proxies of the underlying data patterns and biological events ([:link:](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1893-y)). Therefore, by interpreting the visual concepts, we can inspect how the underlying data and the associated biological events vary among the analyzed items. `dim_7` (renamed as `dim_th`) indicates the thickness of the diagonal (e.g., an item changing from Figure 3A1 to A4), which is related to the size of the TAD since we resize all TADs into a fixed size for the DRL model. Dim_0 indicates the asymmetry of the nested TAD structure (e.g., an item changing from Figure 3A2 to A3). Another meaningful dimension is `dim_6`, which corresponds to whether a TAD data item contains additional nested squares (i.e., nested TAD such as A2, A3) or not (i.e., single TAD such as A1, A4). Other dimensions are either hard to interpret because there is little variation in the synthesized images (`dim_1`, `dim_5`, `dim_4`) or can not be associated with meaningful domain insights (`dim_2`, `dim_3`). `dim_7` and `dim_6` are the top two dimensions based on the salience scores, indicating the usefulness of the dimension ranking. The three dimensions (`dim_th`, `dim_0`, `dim_6`) correspond to important attributes of TADs, as described by [An et al.](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1893-y). 

#### Verify and Refine Concepts
After obtaining a basic understanding of the semantic meaning of each dimension through their synthesized images, we further verify the three concepts one by one through grouping and browsing data items. Interestingly, we find that `dim_6` confuses the thickness of the diagonal with the nested structure of TADs, as shown in Figure 3B. **This issue can hardly be revealed through the synthesized images, which are widely used as the only method to interpret semantic meanings in previous literature ([:link:](https://ieeexplore.ieee.org/abstract/document/9233993))**. The identification of this issue shows the importance of further verifying a concept base on data items and the needs for user refinement.

Since `dim_th` can indicate the TAD size, we use it as the 𝑦 axis to help refine the concept associated with `dim_6`. As shown in Figure 3C, items arranged in different vertical positions based on their diagonal thickness, enabling successfully separation of nested TAD (e.g., A2, A3) from single TADs with thick diagonal (e.g., A4). Users can refine `dim_6` by lasso selection on all single TADs that have large `dim_6` values and moving them to the left-most position (e.g., assigning them a small value for `dim_6`), as shown in Figure 3C1. Since we do not group items based on `dim_6`, the refinement is local and only applied to the user-modified items.

#### Locate items of interest
Users can easily locate nested TADs in Figure 3C2 through a lasso selection. They can also filter these TADs based on `dim_th` and `dim_6` using their histograms. The nested structure in TADs is important to understand the boundary usage in gene regulation ([:link:](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1893-y)). For this purpose, these identified items can be further examined in the Spatial View<!--(Figure 4)-->, which reveals the genomic locations of these TADs and associated them with other context information (e.g., chromatin accessibility).

### Breast Cancer Specimen
<center><img src="https://user-images.githubusercontent.com/9922882/187742532-e6034506-0ca0-4233-af0e-0f52f02ae18c.png" style='max-width: 600px'/></center>

*Fig. 4. (A) Arranging image patches from breast cancer specimens based on concepts learned by Drava shows a strong association between the two visual concepts and the presence of IDC (the color of item labels). We further filter these items to identify confident false-positive predictions (B, C) and locate them in the original whole mount slide image (D).*

#### Data and Model
This usage scenario uses [breast histopathology images from Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images). This dataset contains 277,524 patches (size 50 x 50 pixels) extracted from stained whole mount slide images of breast cancer specimens from 162 patients scanned at 40x magnification. The DRL model is trained on the whole dataset. In this usage scenario, we explore the 1745 image patches from one patient. The DRL model has five convolution blocks, each with a kernel of size 3 and 32, 64, 128, 256, 512 channels, respectively. The latent vector has 12 dimensions.

#### Interpret Visual Concepts and Assign Domain Semantics
We first visualize all the items using a UMAP visualization<!--(Figure 1A)-->. However, the UMAP projection is not ideal since it is based on the overall similarities and considers some irrelevant information, such as the position of tissues patches and the orientation of tissue patches.

Therefore, we check the Concept View to find latent dimensions that can indicate concepts with domain semantics. Based on the synthesized images, we speculate that `dim_5` is related to the density of tissues and `dim_2` is related to the color of the stained tissues. Our interpretation of these two dimensions is further confirmed by examining the grouped items in the Item Browser. <!--As shown in Figure 1B, -->When all items are arranged based on `dim_5`, items on the left side have almost no white space, indicating a high density of tissues, while items on the right side have more white spaces, indicating loose tissues or fatty tissues. When all items are arranged based on `dim_2`, items on the left side have a more purple hue while items on the right side have a more pink hue. We then rename `dim_5` as `dim_density` and `dim_2` as `dim_color`.

We arrange all items using `dim_density` as the 𝑥 axis and `dim_color` as the 𝑦 axis, and then add a label for each item from the item metadata to indicate whether this item contains Invasive Ductal Carcinoma (IDC), a subtype of breast cancer cells. As shown in Figure 4A, there is a strong correlation between the presence of IDC and the two visual concepts. We group items<!--(Figure 1C)--> to reduce the visual clutter. Items with purple and dense tissues (Figure 4A1) are more likely to contain IDC (i.e., orange labels) while items that are more pink (A2) and contain less dense tissue (A3) are less likely to contain IDC (i.e., blue labels). This association is further confirmed by a pathologist. Even though the identification of cancer cells needs to consider a variety of factors, the color and the tissue density are strong indicators of the presence of cancer cells. Cancer cells are typically dense, which leads to less white space, and have larger and darker nucleus than normal cells, which leads to more purple color. 

#### Identify Hard Examples for IDC Identification
Identifying regions in the whole mount slide image (i.e., items in our analysis) with IDC is an important task for pathologists to assign an aggressiveness grade to cancer. Since `dim_dense` and `dim_color` are related to the identification of cancer cells, we further analyzed how they influence the prediction of IDC in a machine learning model. We train a classification model by fine-tuning a ResNet34, as described in ([:link:](https://medium.com/swlh/breast-cancer-classification-with-pytorch-and-deep-learning-52dd62362157)), to predict whether an item contains IDC and record the model prediction and confidence score for each item. 

Confident wrong predictions and false negatives are more consequential in real-world deployment ([:link:](https://ascpt.onlinelibrary.wiley.com/doi/full/10.1111/cts.12478)), as patients may fail to receive the treatment they need. Therefore, we are especially interested in false-negative prediction with high confidence scores. We filter items in the Spatial View accordingly, i.e., ground truth = positive, prediction = negative, confidence score > 0.8, as shown in Figure 4B. According to the Item Browser (Figure 4C), the filtered items are close to each other in the Item Browser, containing tissues that are not very dense and have a more purple hue. Since items with cancer cells usually contain dense tissues, this may explain why the classification model makes a very confident but wrong predictions. We further examine these items in the Spatial View (see Figure Figure 4D), where other items are faded out with a semi-transparent white mask. We find the items of interest (i.e., non-masked items) are from regions where fatty tissues are surrounded by cancer cells, as shown by the orange boxes in Figure 4D. This can explain why these items have many white spaces and only contain a small number of cancer cells.

This observation is valuable for understanding and improving this IDC diagnosis model. First, it indicates when and where the IDC prediction model tends to make confident false negative predictions and a double-check from human experts is needed. Second, the training strategy can be modified accordingly (e.g., increasing the sample weight of these loose and purple tissues) to improve the model performance.

### Datasets Used
- Breast Cancer: https://www.kaggle.com/paultimothymooney/breast-histopathology-images
- Celeba dataset (celebrity faces): https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ 
- Genomic Interaction Data: U54-HFFc6-FA-DSG-MNase-R1-R3.hg38.mapq_30.500.mcool
- Single-Cell Segmentation Masks: [HBM622.JXWQ.554 dataset at HuBMAP data portal](https://portal.hubmapconsortium.org/browse/dataset/13831dc529085f18ba34e7d29bd41db4)
- dSprites (simple Shapes): https://github.com/deepmind/dsprites-dataset