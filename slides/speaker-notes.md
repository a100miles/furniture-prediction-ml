# Furniture listings: type, then price: speaker notes

## Slide 1 (cover)

Sultanbek (about 15 seconds). Introduce the team and the one-line story: the type turned out to be a size question, so the Final’s main question is the price, with the product name as a second source on the same listings.

## Slide 2 (question)

Sultanbek (about 45 seconds). At the Endterm we predicted the type with 0.97. The teacher pointed out that three sizes alone already give 0.92, and we confirmed it on our split: 0.918. The type is a question of shape, so photos could only fix a few border cases. The teacher suggested price on the same listings and test set, and that is our main question now: two sofas of the same size can differ three times in price.

## Slide 3 (existing)

Sultanbek (about 15 seconds). Two price works shaped the method: the IKEA tutorial uses a random forest on sizes and category, which is our specs baseline; the Mercari winners showed that TF-IDF features of product titles carry price, which is what we do with mebel.kz names. The pretrained models are multilingual E5 for text and ResNet-18 for photos.

## Slide 4 (data)

Tamerlan (about 45 seconds). The store’s copyright notice covers photos and page text, so we use neither. Product names are short factual labels on exactly the same listings, so they are our second source on the same test objects. Amazon Berkeley Objects is CC BY 4.0, but other products without price, so it is only for the type. The split is unchanged since the Midterm: 793 test listings, 92 designs, colour variants kept together, seed 42.

## Slide 5 (models)

Sultanbek (about 30 seconds). All models on one slide. Type: seven classifiers, the SVM won in cross-validation. Price: six regressors on the specifications, LightGBM best; with the name, our own TF-IDF features against a pretrained multilingual encoder, and an SVR on TF-IDF won. Photos: our small CNN against frozen ResNet-18, CLIP and DINOv2, plus a fine-tuned ResNet-18; DINOv2 won.

## Slide 6 (method)

Tamerlan (about 30 seconds). One recipe for every model: scaled sizes, one-hot colour and type, log price, the name as TF-IDF or embeddings, a photo as a frozen network’s vector. The 793 test listings are locked away and a design never crosses the split. Settings are chosen by grouped 5-fold CV inside one pipeline, and the winner runs on the test set once.

## Slide 7 (type)

Sultanbek (about 1 minute). The Endterm feedback asked us to check colour variants: 44 test listings had a twin in training. With a stricter grouping the random forest fell from 0.958 to 0.933, and the SVM, 0.949, became the chosen model; per design the test score is 0.940. Coffee tables are only 4% but have PR-AUC 0.991; the weak classes are TV stands, desks and chests. And the model does not travel: 0.606 on Amazon listings.

## Slide 8 (price)

Sultanbek (about 1 minute 30 seconds). Same listings, same 793 test listings, same groups. The median price of the type is off by 95 thousand tenge; specifications bring it to about 61 thousand, and all tree, linear and kernel models, LightGBM included, are within the fold spread. Adding the name with our own TF-IDF features and an SVR gives 59 thousand in CV and 51.7 thousand on the test set, MAPE 20.6 versus 22.1 percent. The pretrained e5 encoder adds less: names are short and full of shop words. On IKEA Saudi Arabia the name also helps, from 47 to 40 percent error.

## Slide 9 (check)

Tamerlan (about 45 seconds). Two checks. First leakage: the name contains the design’s own name, which could give away its price. Our groups keep a design on one side of every split, and on top we masked the design name: three quarters of the gain stays. Second, run-to-run variation: with five other fold cuts, the name model wins in all six, by 1.5 to 4.3 thousand tenge over LightGBM. The gain is small, about 2 to 7 percent of the error, but it is not an accident of one split.

## Slide 10 (errors)

Tamerlan (about 30 seconds). The name helps where the specs miss the function: sliding-door wardrobes, wall-hung TV stands, boucle fabric. It does not fix modular sofas, which are our largest errors, because neither the sizes nor the name tell the model the configuration. Chests look worse with the name, but that is one design in sixteen colour and pattern variants.

## Slide 11 (photos)

Tamerlan (about 45 seconds). mebel.kz photos are not allowed, so we tested the photo idea on Amazon Berkeley Objects. On the same listings a frozen ResNet-18 beats the specifications, 0.766 versus 0.662, while our own small CNN underfits at 0.605. A stronger pretrained encoder helped most: DINOv2 reaches 0.844, 0.851 with the specifications, and TV stands go from 8 to 78 percent found. This is on other products, without price.

## Slide 12 (demo)

Tamerlan (about 1 minute). Start the server before the talk (python api.py) and keep the page open. Price mode: load the Sofa example, predict without a name, then with the name; point at the error and the 8-in-10 range. Type a width of 2000 and show the error message. Switch to From a photo, drop one chair photo, predict. If the demo fails, go back to the price slide and say what the app would show.

## Slide 13 (conclusions)

Sultanbek (about 45 seconds). Specifications predict the price within about a quarter; the name helps a little but steadily, not by memorising designs, and it does so in a second market too. Our own TF-IDF features beat the pretrained encoder on these short names. The type is a size question and the model stays in its store. For photos, picking a stronger pretrained model mattered more than training our own. Limits: one store and one day, text instead of photos, CV slightly optimistic and a small test set of 92 designs. Thank you.
