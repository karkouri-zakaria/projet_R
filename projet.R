# Charger les données des clients depuis les fichiers CSV
clients <- read.csv("temp/Data Projet.csv", header = TRUE, sep = ",", dec = ".", stringsAsFactors = TRUE)
clients_New <- read.csv("temp/Data Projet New.csv", header = TRUE, sep = ",", dec = ".", stringsAsFactors = TRUE)

# Afficher les premières lignes, la structure et le résumé des données
head(clients)
str(clients)
summary(clients)

# Afficher la table des fréquences de la variable 'default'
table(clients$default)

# Supprimer les colonnes inutiles et les valeurs manquantes
clients <- clients[, !(names(clients) %in% c('customer', 'ncust'))]
clients <- na.omit(clients)

# Séparer les données en ensembles d'apprentissage (EA) et de test (ET)
clients_EA <- clients[1:800, ]
clients_ET <- clients[801:nrow(clients), ]
clients_EA <- clients_EA[, -1]

# Construire les modèles d'arbre de décision avec différentes configurations
tree_rpart1 <- rpart(default ~ ., data = clients_EA, method = "class", control = rpart.control(split = "gini", minbucket = 10))
tree_rpart2 <- rpart(default ~ ., data = clients_EA, method = "class", control = rpart.control(cp = 0, minsplit = 2, minbucket = 5))
tree_rpart3 <- rpart(default ~ ., data = clients_EA, method = "class", control = rpart.control(cp = 0, minsplit = 2, minbucket = 10, split = "information"))
tree_rpart4 <- rpart(default ~ ., data = clients_EA, method = "class", control = rpart.control(cp = 0, minsplit = 2, minbucket = 5, split = "information"))
tree_c50_1 <- C5.0(clients_EA[, -ncol(clients_EA)], clients_EA$default, control = C5.0Control(minCases = 10, noGlobalPruning = FALSE))
tree_c50_2 <- C5.0(clients_EA[, -ncol(clients_EA)], clients_EA$default, control = C5.0Control(minCases = 10, noGlobalPruning = TRUE))
tree_c50_3 <- C5.0(clients_EA[, -ncol(clients_EA)], clients_EA$default, control = C5.0Control(minCases = 5, noGlobalPruning = FALSE))
tree_c50_4 <- C5.0(clients_EA[, -ncol(clients_EA)], clients_EA$default, control = C5.0Control(minCases = 5, noGlobalPruning = TRUE))
tree_tree_1 <- tree(default ~ ., data = clients_EA, split = "deviance", mincut = 10)
tree_tree_2 <- tree(default ~ ., data = clients_EA, split = "deviance", mincut = 5)
tree_tree_3 <- tree(default ~ ., data = clients_EA, split = "gini", mincut = 10)
tree_tree_4 <- tree(default ~ ., data = clients_EA, split = "gini", mincut = 5)

# Faire des prédictions sur l'ensemble de test avec chaque modèle
predictions_rpart1 <- predict(tree_rpart1, newdata = clients_ET, type = "class")
predictions_rpart2 <- predict(tree_rpart2, newdata = clients_ET, type = "class")
predictions_rpart3 <- predict(tree_rpart3, newdata = clients_ET, type = "class")
predictions_rpart4 <- predict(tree_rpart4, newdata = clients_ET, type = "class")
predictions_c50_1 <- predict(tree_c50_1, newdata = clients_ET)
predictions_c50_2 <- predict(tree_c50_2, newdata = clients_ET)
predictions_c50_3 <- predict(tree_c50_3, newdata = clients_ET)
predictions_c50_4 <- predict(tree_c50_4, newdata = clients_ET)
predictions_tree_1 <- predict(tree_tree_1, newdata = clients_ET, type = "class")
predictions_tree_2 <- predict(tree_tree_2, newdata = clients_ET, type = "class")
predictions_tree_3 <- predict(tree_tree_3, newdata = clients_ET, type = "class")
predictions_tree_4 <- predict(tree_tree_4, newdata = clients_ET, type = "class")

# Définir une fonction pour calculer le ratio de prédictions correctes
calculate_ratio <- function(actual, predicted) mean(actual == predicted)

# Calculer les ratios de prédictions correctes pour chaque modèle
ratio_rpart1 <- calculate_ratio(clients_ET$default, predictions_rpart1)
ratio_rpart2 <- calculate_ratio(clients_ET$default, predictions_rpart2)
ratio_rpart3 <- calculate_ratio(clients_ET$default, predictions_rpart3)
ratio_rpart4 <- calculate_ratio(clients_ET$default, predictions_rpart4)
ratio_c50_1 <- calculate_ratio(clients_ET$default, predictions_c50_1)
ratio_c50_2 <- calculate_ratio(clients_ET$default, predictions_c50_2)
ratio_c50_3 <- calculate_ratio(clients_ET$default, predictions_c50_3)
ratio_c50_4 <- calculate_ratio(clients_ET$default, predictions_c50_4)
ratio_tree_1 <- calculate_ratio(clients_ET$default, predictions_tree_1)
ratio_tree_2 <- calculate_ratio(clients_ET$default, predictions_tree_2)
ratio_tree_3 <- calculate_ratio(clients_ET$default, predictions_tree_3)
ratio_tree_4 <- calculate_ratio(clients_ET$default, predictions_tree_4)

# Créer un data frame avec les résultats des prédictions pour les nouveaux clients
result_df <- data.frame(
  customer_id = clients_New$customer,
  predicted_class = predict(tree_tree_2, newdata = clients_New, type = "class"),
  predicted_prob = apply(predict(tree_tree_2, newdata = clients_New, type = "vector"), 1, max)
)

# Afficher un résumé du data frame résultant
summary(result_df)

# Sauvegarder les résultats des prédictions dans un fichier CSV
write.csv(result_df, "temp/Prediction_results.csv", row.names = FALSE)