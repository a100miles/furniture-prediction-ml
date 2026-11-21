# Furniture listings: type, then price: speaker notes

## Slide 1 (cover)

Sultanbek (about 15 seconds). Introduce the team and the one-line story: the type turned out to be a size question, so the Final’s main question is the price, with the product name as a second source on the same listings.

## Slide 2 (question)

Sultanbek (about 1 minute). At the Endterm we predicted the type with 0.97. The teacher pointed out that three sizes alone already give 0.92, and we confirmed it on our own split: 0.918. A sofa and a wardrobe are told apart by their shape, so photos could only fix a few border cases. The teacher suggested price on the same listings and test set, and that is our main question now. Price is different: two sofas of the same size can differ three times in price.

## Slide 3 (existing)

Sultanbek (about 15 seconds). Two price works shaped the method: the IKEA tutorial uses a random forest on sizes and category, which is our specs baseline; the Mercari winners showed that TF-IDF features of product titles carry price, which is what we do with mebel.kz names. The pretrained models are multilingual E5 for text and ResNet-18 for photos.

## Slide 4 (data)

Tamerlan (about 1 minute). The store’s copyright notice covers photos and page text, so we use neither. Product names are short factual labels, like the sizes, and they belong to exactly the same listings, so they give us a second source on the same test objects. Amazon Berkeley Objects is CC BY 4.0, but it is other products without price, so we use it only for the type. The split is the same as since the Midterm: 793 test listings, 92 designs, groups keep colour variants together, seed 42.

## Slide 5 (type)

Sultanbek (about 1 minute 15 seconds). The Endterm feedback asked us to check colour variants. Our grouping missed 44 test listings whose twin was in training. With a stricter grouping the random forest fell from 0.958 to 0.933 and the SVM, 0.949, became the chosen model. Counting each design once, the test score is 0.940. Coffee tables are only 4% but have PR-AUC 0.991; the weak classes are TV stands, desks and chests. And the model does not travel: on Amazon listings it scores 0.606.

## Slide 6 (price)

Sultanbek (about 1 minute 45 seconds). Same 3,961 listings, same 793 test listings, same groups. The median price of the type gives an error of 95 thousand tenge. Specifications bring it to about 61 thousand: a random forest, boosting, SVR and ridge are all within the fold spread. Adding the name with our own TF-IDF features and an SVR gives 59 thousand in CV and 51.7 thousand on the test set, MAPE 20.6 versus 22.1 percent. The pretrained multilingual e5 encoder adds less than our own features: names are short and full of shop-specific words like fabric names and sizes, which character pieces catch directly.

## Slide 7 (check)

Tamerlan (about 45 seconds). Two checks. First leakage: the name contains the design’s own name, which could give away its price. Our groups keep a design on one side of every split, and on top we masked the design name: three quarters of the gain stays. Second, run-to-run variation: with five other fold cuts, the name model wins in all six, by 2.4 to 4.1 thousand tenge. The gain is small, about 4 to 6 percent of the error, but it is not an accident of one split.

## Slide 8 (errors)

Tamerlan (about 30 seconds). The name helps where the specs miss the function: sliding-door wardrobes, wall-hung TV stands, boucle fabric. It does not fix modular sofas, which are our largest errors, because neither the sizes nor the name tell the model the configuration. Chests look worse with the name, but that is one design in sixteen colour and pattern variants.

## Slide 9 (photos)

Tamerlan (about 45 seconds). Our Endterm plan was photos for the case furniture. mebel.kz photos are not allowed, so we tested the idea on Amazon Berkeley Objects, which is CC BY 4.0. On the same Amazon listings, a frozen ResNet-18 beats the specifications, 0.766 versus 0.662, and TV stands go from almost never found to mostly found. Our own small CNN, trained from scratch on about 2,700 products, underfits at 0.605. This is on other products, without price.

## Slide 10 (demo)

Tamerlan (about 1 minute). Open the Predict page. Price mode: load the Sofa example, predict without a name, then type the name and predict again; point at the range and the CV error. Then type a width of 2000 and show the error message. Finally switch to From a photo, drop one chair photo, predict. Start the server before the talk (python api.py) and keep the page open in a tab. If the demo fails, go back to the price slide and say what the app would show.

## Slide 11 (conclusions)

Sultanbek (about 45 seconds). Specifications predict the price within about a quarter; the name helps a little but steadily, and not by memorising designs. Our own TF-IDF features beat the pretrained encoder on these short names. The type is a size question and the model stays in its store. Limits: one store and one day, text instead of photos, CV slightly optimistic and a small test set of 92 designs. Thank you.
