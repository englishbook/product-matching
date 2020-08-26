# product-matching
The code of Team Rhinobird for [Mining the Web of HTML-embedded Product Data Task One at ISWC2020](https://ir-ischool-uos.github.io/mwpd/index.html#data1).

Task one: Product Matching

The product matching task aims to identify that if a pair of product deriving from different websites refer to the same product or not. 


## Datasets
In the SWC2020 challenge product matching task, the dataset of Task one is sampled from the WDC product data corpus. Products in the corpus are described by these properties: id, cluster id, category, title, description, brand, price, and specification table. 
Our models are mainly trained on two different matching dataset:

- [Computers dataset](https://ir-ischool-uos.github.io/mwpd/index.html#data1) is provided by the organizers of the challenge which only contains product from Computers & Accessories.

- [All dataset](http://webdatacommons.org/largescaleproductcorpus/v2/index.html) contains products from all the four categories (Computers & Accessories, Camera & Photo, Watches, and Shoes).


## Input
Although products are described by many attributes, most of the fields contain NULL values.
Considering the filling rate and the input length, we focus on the title and description attributes and ignore the other ones.


## Model
We use [BERT_base](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) as the main module of our matching model. 
Focal loss is adopted to alleviate class imbalance problem. 

Please download the dataset and BERT weights first.

Just run the train.py to train all the models we used in the challenge:
```python
python train.py
```

After obtaining the model parameters, run the predict.py to combine the predictions of different model and get the final results:
```python
python predict.py
```

## Post-processing
For test pairs with prediction results of 1 but different categories, we directly correct their results to 0 in the post-processing phase.


## Results
### Validation
Single model:

|    Model   | Input | Dataset |   F1   | Post F1 |
|:----------:|:-----:|:-------:|:------:|:-------:|
| Bert_focal | title |   All   | 0.9481 |  0.9496 |
| Bert_focal | title+description |   All   | 0.9384 |  0.9411 |
| Bert_focal | title+description |Computers| 0.9700 |  0.9700 |

### Test
In the final evaluation, we ensemble these three models:

|    Model   | Precision | Recall |   F1   |
|:----------:|:---------:|:------:|:------:|
| Our model  |   0.8063  | 0.9200 | 0.8594 |


