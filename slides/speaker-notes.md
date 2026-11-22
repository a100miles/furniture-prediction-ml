# Furniture listings: type, then price: speaker notes

## Slide 1 (cover)

Sultanbek (about 15 seconds). Introduce the team and the one-line story: the type turned out to be a size question, so the Final’s main question is the price, with the product name as a second source on the same listings.

## Slide 2 (question)

Sultanbek (about 1 minute). At the Endterm we predicted the type with 0.97. The teacher pointed out that three sizes alone already give 0.92, and we confirmed it on our own split: 0.918. A sofa and a wardrobe are told apart by their shape, so photos could only fix a few border cases. The teacher suggested price on the same listings and test set, and that is our main question now. Price is different: two sofas of the same size can differ three times in price.

## Slide 3 (existing)

Sultanbek (about 15 seconds). Two price works shaped the method: the IKEA tutorial uses a random forest on sizes and category, which is our specs baseline; the Mercari winners showed that TF-IDF features of product titles carry price, which is what we do with mebel.kz names. The pretrained models are multilingual E5 for text and ResNet-18 for photos.

## Slide 4 (data)

Tamerlan (about 1 minute). The store’s copyright notice covers photos and page text, so we use neither. Product names are short factual labels, like the sizes, and they belong to exactly the same listings, so they give us a second source on the same test objects. Amazon Berkeley Objects is CC BY 4.0, but it is other products without price, so we use it only for the type. The split is the same as since the Midterm: 793 test listings, 92 designs, groups keep colour variants together, seed 42.

## Slide 5 (models)

Sultanbek (about 45 seconds). This is the whole model zoo on one slide. For the type we compared seven classifiers, from a dummy baseline and a decision tree to an SVM, random forest, gradient boosting and a small neural network; the SVM won in cross-validation. For the price, six regressors on the specifications; LightGBM, a boosted-trees library, was best. Then the same regressors with the product name, either as our own TF-IDF word and character pieces or as vectors from a pretrained multilingual encoder; an SVR on our TF-IDF features won. For photos, a small CNN we trained ourselves against three pretrained networks, ResNet-18, CLIP and DINOv2, used frozen with a logistic regression on top, plus a fine-tuned ResNet-18. DINOv2 won.

## Slide 6 (method)

Tamerlan (about 40 seconds). Every model goes through the same recipe. Sizes are scaled, colour and type one-hot encoded, the price is predicted on a log scale, the name becomes TF-IDF or embedding vectors and a photo becomes the vector of a frozen network. The 793 test listings are locked away, and all colours of one design stay on the same side. Settings are searched with grouped 5-fold cross-validation inside one pipeline, the winner is picked by its CV mean and runs on the test set once. Then we check per-class scores, leakage through design names, other fold seeds and other stores.

## Slide 7 (type)

Sultanbek (about 1 minute 15 seconds). The Endterm feedback asked us to check colour variants. Our grouping missed 44 test listings whose twin was in training. With a stricter grouping the random forest fell from 0.958 to 0.933 and the SVM, 0.949, became the chosen model. Counting each design once, the test score is 0.940. Coffee tables are only 4% but have PR-AUC 0.991; the weak classes are TV stands, desks and chests. And the model does not travel: on Amazon listings it scores 0.606.

## Slide 8 (price)

Sultanbek (about 1 minute 45 seconds). Same 3,961 listings, same 793 test listings, same groups. The median price of the type gives an error of 95 thousand tenge. Specifications bring it to about 61 thousand; after the survey we added LightGBM, which is the best specifications model, but only by a hair: all tree, linear and kernel models are within the fold spread. Adding the name with our own TF-IDF features and an SVR gives 59 thousand in CV and 51.7 thousand on the test set, MAPE 20.6 versus 22.1 percent. The pretrained e5 encoder adds less than our own features: names are short and full of shop words and sizes. And the same holds in a second market: on IKEA Saudi Arabia the name brings the error from 47 to 40 percent.

## Slide 9 (check)

Tamerlan (about 45 seconds). Two checks. First leakage: the name contains the design’s own name, which could give away its price. Our groups keep a design on one side of every split, and on top we masked the design name: three quarters of the gain stays. Second, run-to-run variation: with five other fold cuts, the name model wins in all six, by 1.5 to 4.3 thousand tenge over LightGBM. The gain is small, about 2 to 7 percent of the error, but it is not an accident of one split.

## Slide 10 (errors)

Tamerlan (about 30 seconds). The name helps where the specs miss the function: sliding-door wardrobes, wall-hung TV stands, boucle fabric. It does not fix modular sofas, which are our largest errors, because neither the sizes nor the name tell the model the configuration. Chests look worse with the name, but that is one design in sixteen colour and pattern variants.

## Slide 11 (photos)

Tamerlan (about 1 minute). Our Endterm plan was photos for the case furniture. mebel.kz photos are not allowed, so we tested the idea on Amazon Berkeley Objects, which is CC BY 4.0. On the same Amazon listings, a frozen ResNet-18 beats the specifications, 0.766 versus 0.662. Our own small CNN, trained from scratch, underfits at 0.605. After the survey we tried stronger models on the same folds: fine-tuning ResNet-18 gives 0.789, CLIP 0.801, and a frozen DINOv2 0.844, or 0.851 with the specifications; TV stands go from 8 to 78 percent found. A better pretrained encoder helped more than training a weaker one further. DINOv2 also recognised IKEA photos it had never seen, 0.880. This is on other products, without price.

## Slide 12 (demo)

Tamerlan (about 1 minute). Open the Predict page. Price mode: load the Sofa example, predict without a name, then type the name and predict again; point at the plus-minus error and the 8-in-10 range. Type a made-up name like asdkjh and show that the app falls back to the specifications and says why. Then From specifications: switch off the price and predict; type a width of 2000 and show the error message. Switch to From a photo, drop one chair photo, predict. If time allows, open How it works for one second. Start the server before the talk (python api.py) and keep the page open in a tab. If the demo fails, go back to the price slide and say what the app would show.

## Slide 13 (conclusions)

Sultanbek (about 45 seconds). Specifications predict the price within about a quarter; the name helps a little but steadily, not by memorising designs, and it does so in a second market too. Our own TF-IDF features beat the pretrained encoder on these short names. The type is a size question and the model stays in its store. For photos, picking a stronger pretrained model mattered more than training our own. Limits: one store and one day, text instead of photos, CV slightly optimistic and a small test set of 92 designs. Thank you.
