- Add pytorch lightning integration tests
- ajouter le distribué - Now
- Create attributes that return the last precomputed metric - V1
- Can our method function with a different number of images foe each class? If not, how to handle that case? NOW
- Define the aggreg and distance method in the computing itself - V1
- Add different correlation metrics for interclass
- Find api for using cell profiler on test when asked by user
- Revamp library to be fully compatible with torchmetrics too (Maybe? maybe not necessary)
- Test metrics with different code design, to see most adaapted for usage in pytorch and PL
- possibility to use the metric as a callback in PL
- possibility to use the metric on a data subset only
- Multiprocessing? both cpu and gpu. It should be both adaptive to pytorch and PL multiprocessing, or do multiprocessing for acceleration on its own
- Add wiki files that explain usage and eaxmples and docs of the code
- Automate github actions to test automatically the code in containers after each push
- Add a wiki page for the project


--- Distribued tasks :
- Make intra-class distributed distance metric
- Add interclass distrubuted metric
- Integrate into main function
- Add unit tests for distributed metrics
- Add FID metric
- Add KID metric
- Add IS metric
- Add intra-class FID and KID metric
- Make functions for reading images of npz files or folders
- Make a function to test all metrics on a given dataset


--- Control tasks :
- Write better code documentation after control additions
- Note that oder of classes and number of classes in input is important to keep as the full classes of data
- prepare the readme and tutorials for usage of the code
- Add license file
- Link contributing file and license file to main readme
- Add a code of conduct file
- Add a pull request template file
- Add a issue template file
- Write a wiki page describing the metric class and its usage
