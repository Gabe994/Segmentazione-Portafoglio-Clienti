# FUNZIONI
#####
##FUNZIONE PER SETTARE MINSPLIT E CP CON CARET (CUSTOM TREE)

customTree <- list(type = "Classification", library = "rpart", loop = NULL)
head(customTree)
customTree$parameters <- data.frame(parameter = c("cp", "minsplit"), class = rep("numeric", 2), label = c("cp", "minsplit"))
customTree$grid <- function(x, y, len = NULL, search = "grid") {}
customTree$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, cp = param$cp, minsplit=param$minsplit, mtry=22, ntree=1, ...)
}
# facciamo random forest usando tutte le variabili (mtry=22) e un solo albero per trovare tutti gli alberi possibili con questi min split e cp
# random forest ti trova già l'albero migliore

# customize our predicted function
customTree$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customTree$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customTree$sort <- function(x) x[order(x[,1]),]
customTree$levels <- function(x) x$classes

Mode <- function(x) { # CALCOLARE LA MODA
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#####
# LIBRARY
#####
library(funModeling)
library(dplyr)
library(AER) # DurbinWatson Test
library(reshape2) # funzione melt
library(car) # verificare multicollinearità
library(gvlma) # verifica assunzioni glm
library(broom) # # distanza di cook
library(rpart) # tree
library(rpart.plot) # plot tree
library(caret) # tuning
library(randomForest)
library(leaps) # feature selection con regsubsets
library(Hmisc) # per usare describe
library(ggplot2)
library(cluster)
library(gvlma) # per controllare ipotesi regression
options(scipen=999)

#####
# CARICO E SISTEMO DATASET
#####
setwd('C:/Users/hp/Google Drive/Università/Data Science Lab/Progetto')
dati_clienti <- read.csv("DatasetClientClustering.csv",encoding='UTF-8')
prov = read.csv("Provincia.csv", sep = ";")
descrizionecluster = dati_clienti[0:25, 0:2]
dati_clienti=dati_clienti[,9:33] #rimuovo attributi in più
dati_clienti <- dati_clienti %>% mutate(DurataRapporto = 2018 - ClientDateStart) # creo colonna durata rapporto
summary(dati_clienti$DurataRapporto)
dati_clienti$ClientDateStart <- NULL
status = df_status(dati_clienti, print_results = F)
dati_clienti$IncomeHighLow=as.factor(dati_clienti$IncomeHighLow)
dati_clienti$Sex=as.factor(dati_clienti$Sex)
# problemi con Panic Mood (valori -1 e 0).converto in 1,0
table(dati_clienti$PanicMood)
dati_clienti$PanicMood <- as.numeric(dati_clienti$PanicMood)
dati_clienti$PanicMood[dati_clienti$PanicMood == -1] <- 1
dati_clienti$PanicMood <- as.factor(dati_clienti$PanicMood)
dati_clienti$NoTrustInBanks=as.factor(dati_clienti$NoTrustInBanks)
status = df_status(dati_clienti, print_results = F)
Quant <- filter(status,  type %in% c("numeric", "integer")) %>% .$variable
dati_clientiQ <- select(dati_clienti, one_of(Quant))
dati_clienti_NOPortfolio <- dati_clienti[, -c(1, 3, 5, 19:24)]

#####
# CLUSTER MFID
#####

#IPOTESI DI SEGMENTAZIONE:
#clienti che rischiano tanto/clienti che rischiano poco (variabile: RiskPropension)
#clienti breve periodo/lungo periodo (variabile: ClientInvestmentHorizon)
#conoscenza finanziaria del cliente (variabile: ClientKnowledgeExperience)

dati_clienti_persona=dati_clienti[,c(2,4,6,12,13,14,15,16)] #SOLO I DATI RIGUARDANTI LA PERSONALITA' DEL CLIENTE, NON IL SUO PORTAFOGLIO

library(corrplot)
M <- cor(dati_clienti_persona)
corrplot.mixed(M, lower = "number", upper="ellipse",tl.pos = "lt", tl.col="black")
#NESSUNA CORRELAZIONE ECCESSIVAMENTE ALTA (SAREBBE DA VERIFICARE CON MULTICOLLINEARITA') QUINDI POSSIAMO ANCHE NON RIMUOVERE ATTRIBUTI

summary(dati_clienti_persona)
#Hanno diverse unità di misura, serve quindi standardizzare
dati_clienti_persona_stand=as.data.frame(scale(dati_clienti_persona))


km.out <- list()
sil.out <- list()
x <- vector()
y <- vector()
minClust <- 2     
maxClust <- 10

set.seed(592)
for (centr in minClust:maxClust) {
  i <- centr-(minClust-1) 
  km.out[i] <- list(kmeans(dati_clienti_persona_stand, centers = centr, nstart = 50,iter.max = 50))
  sil.out[i] <- list(silhouette(km.out[[i]][[1]], dist(dati_clienti_persona_stand)))
  x[i] = centr  # value of k
  y[i] = summary(sil.out[[i]])[[4]]  # Silhouette average width
}

ggplot(data = data.frame(x, y), aes(x, y)) + 
  geom_point(size=3) + 
  geom_line() +
  xlab("Number of Cluster Centers") +
  ylab("Silhouette Average Width") +
  ggtitle("Silhouette Average Width as Cluster Center Varies")

ks <- 2:10 # number of clusters we want to try
ssw <- numeric(length(ks)) # vector for the ss_within
for (i in seq_along(ks)) {
  ssw[i] <- kmeans(dati_clienti_persona_stand, ks[i],iter.max = 50)$tot.withinss
}

plot(x = ks, y = ssw, type = "l",
     xlab = "Number of clusters",
     ylab = "SS_within",
     main = "Look for an elbow")
plot(x = ks[-1],
     y = - ssw[-1] + ssw[-length(ssw)], type = "h",
     xlab = "Number of clusters",
     ylab = "Decrement in SS_within",
     main = "Look for a spike")


#SCEGLIAMO DI UTILIZZARE 3 CLUSTER
km.out.best <- km.out[[2]] 

#GUARDIAMO I CENTROIDI DEI CLUSTER PER CAPIRE IN COSA SI DISTINGUONO RISPETTO ALLE VARIBILI UTLIZZATE PER CLUSTERIZZARE:

custSegmentCntrs <- t(km.out.best$centers)  # Get centroids for groups
colnames(custSegmentCntrs) <- make.names(colnames(custSegmentCntrs))
custSegmentCntrs

#INTEGRIAMO CON ANALISI DELLE COMPONENTI PRINCIPALI

pca <- prcomp(dati_clienti_persona, scale. = T, center = T)
summary(pca)

library(ggfortify)
pca.fortify <- fortify(pca)

pca3.dat <- cbind(pca.fortify, group=km.out.best$cluster)

library(ggplot2)
gg2 <- ggplot(pca3.dat) +
  geom_point(aes(x=PC1, y=PC2, col=factor(group), text=rownames(pca3.dat)), size=2) +
  labs(title = "Visualizing K-Means Clusters Against First Two Principal Components") +
  scale_color_brewer(name="", palette = "Set1")

plot(gg2)

#VEDIAMO I PESI DELLE VARIABILI SULLE PRIME DUE CP

theta <- seq(0,2*pi,length.out = 100)
circle <- data.frame(x = cos(theta), y = sin(theta))
p <- ggplot(circle,aes(x,y)) + geom_path()

loadings <- data.frame(pca$rotation, 
                       .names = row.names(pca$rotation))
p + geom_text(data=loadings, 
              mapping=aes(x = PC1, y = PC2, label = .names, colour = .names)) +
  coord_fixed(ratio=1) +
  labs(x = "PC1", y = "PC2", cex=10)


#INTERPRETARE I CLUSTER SULLA BASE DEI PESI SULLE CP (SOPRATTUTTO DELLA PRIMA CP)

#####
# CLUSTER PORTAFOGLI
#####

#PROVIAMO ADESSO A CLUSTERIZZARE IN BASE AL PORTAFOGLIO E NON IN BASE ALLA PERSONALITA'

dati_clienti_portafoglio=dati_clienti[,c(20,21,22,23,24)]

library(corrplot)
M1 <- cor(dati_clienti_portafoglio)
corrplot.mixed(M1, lower = "number", upper="ellipse",tl.pos = "lt",tl.col="black")
#NESSUNA CORRELAZIONE ECCESSIVAMENTE ALTA (SAREBBE DA VERIFICARE CON MULTICOLLINEARITA') QUINDI POSSIAMO ANCHE NON RIMUOVERE ATTRIBUTI
#ANCHE SE SI AVRà CHE COMUNQUE QUESTA COMBINAZIONE DI ATTRIBUTI SARà LINEARMENTE DIPENDENTE (ESSENDO PARTI DI UN TOTALE CHE SAPPIAMO ESSERE 1)

summary(dati_clienti_portafoglio)
#NON SERVE STANDARDIZZARE, HANNO UGUALE UNITà DI MISURA


km.out1 <- list()
sil.out1 <- list()
x1 <- vector()
y1 <- vector()
minClust1 <- 2     
maxClust1 <- 10

set.seed(564)
for (centr in minClust1:maxClust1) {
  i <- centr-(minClust1-1) 
  km.out1[i] <- list(kmeans(dati_clienti_portafoglio, centers = centr, nstart = 50,iter.max = 50))
  sil.out1[i] <- list(silhouette(km.out1[[i]][[1]], dist(dati_clienti_portafoglio)))
  x1[i] = centr  # value of k
  y1[i] = summary(sil.out1[[i]])[[4]]  # Silhouette average width
}


ggplot(data = data.frame(x1, y1), aes(x1, y1)) + 
  geom_point(size=3) + 
  geom_line() +
  xlab("Number of Cluster Centers") +
  ylab("Silhouette Average Width") +
  ggtitle("Silhouette Average Width as Cluster Center Varies")

ks1 <- 2:10 # number of clusters we want to try
ssw1 <- numeric(length(ks1)) # vector for the ss_within
for (i in seq_along(ks1)) {
  ssw1[i] <- kmeans(dati_clienti_portafoglio, ks1[i],iter.max = 50)$tot.withinss
}

plot(x = ks1, y = ssw1, type = "l",
     xlab = "Number of clusters",
     ylab = "SS_within",
     main = "Look for an elbow")
plot(x = ks1[-1],
     y = - ssw1[-1] + ssw1[-length(ssw1)], type = "h",
     xlab = "Number of clusters",
     ylab = "Decrement in SS_within",
     main = "Look for a spike")


#SCEGLIAMO DI UTILIZZARE 3 CLUSTER
km1.out.best <- km.out1[[2]] 

#GUARDIAMO I CENTROIDI DEI CLUSTER PER CAPIRE IN COSA SI DISTINGUONO RISPETTO ALLE VARIBILI UTLIZZATE PER CLUSTERIZZARE:

custSegmentCntrs1 <- t(km1.out.best$centers)  # Get centroids for groups
colnames(custSegmentCntrs1) <- make.names(colnames(custSegmentCntrs1))
custSegmentCntrs1

#INTEGRIAMO CON ANALISI DELLE COMPONENTI PRINCIPALI

pca1 <- prcomp(dati_clienti_portafoglio, scale. = T, center = T)
summary(pca1)

library(ggfortify)
pca.fortify1 <- fortify(pca1)

pca3.dat1 <- cbind(pca.fortify1, group=km1.out.best$cluster)


gg2 <- ggplot(pca3.dat1) +
  geom_point(aes(x=PC1, y=PC2, col=factor(group), text=rownames(pca3.dat1)), size=2) +
  labs(title = "Visualizing K-Means Clusters Against First Two Principal Components") +
  scale_color_brewer(name="", palette = "Set1")

plot(gg2)

#VEDIAMO I PESI DELLE VARIABILI SULLE PRIME DUE CP

theta <- seq(0,2*pi,length.out = 100)
circle <- data.frame(x = cos(theta), y = sin(theta))
p1 <- ggplot(circle,aes(x,y)) + geom_path()

loadings1 <- data.frame(pca1$rotation, 
                        .names = row.names(pca1$rotation))
p1 + geom_text(data=loadings1, 
               mapping=aes(x = PC1, y = PC2, label = .names, colour = .names)) +
  coord_fixed(ratio=1) +
  labs(x = "PC1", y = "PC2")


#INTERPRETARE ANCHE QUI LE INFORMAZIONI

#####
# UNIONE DEI CLUSTER
#####

###Unire i due cluster

custSegmentCntrs
custSegmentCntrs1

dataset_finale <- cbind(dati_clienti_persona, ClusterProfili = km.out.best$cluster, ClusterPortafoglio = km1.out.best$cluster)

dataset_finale$Match <- ifelse((dataset_finale$ClusterProfili == 1 & dataset_finale$ClusterPortafoglio == 3) |
                                 (dataset_finale$ClusterProfili == 2 & dataset_finale$ClusterPortafoglio == 2) | 
                                 (dataset_finale$ClusterProfili == 3 & dataset_finale$ClusterPortafoglio == 1),
                               TRUE, FALSE)
status_final <- df_status(dataset_finale)
dataset_finale$ClusterPortafoglio <- as.factor(dataset_finale$ClusterPortafoglio)
dataset_finale$ClusterProfili <- as.factor(dataset_finale$ClusterProfili)
status_final <- df_status(dataset_finale)

# Divido in test e train
dataset_finale_training <- dataset_finale[which(dataset_finale$Match == TRUE), ]
dataset_finale_test <- dataset_finale[which(dataset_finale$Match == FALSE), ]

# PROVO MODELLI PER TROVARE QUELLO PIù PERFORMANTE

dataClusterTrain <- dataset_finale_training
dataClusterTrain$ClusterPortafoglio <- make.names(dataClusterTrain$ClusterPortafoglio)
dataClusterTrain$ClusterPortafoglio <- as.factor(dataClusterTrain$ClusterPortafoglio)
str(dataClusterTrain)


# Rpart

set.seed(7)
tunegrid <- expand.grid(cp=seq(0.005, 0.1, 0.001))
treetuneClusterrpart <- train(ClusterPortafoglio ~ . - Match - ClusterProfili, data=dataClusterTrain, method='rpart', tuneGrid=tunegrid,
                              metric='Accuracy')
treetuneClusterrpart # cp 0.007 e minsplit = 20 di default
getTrainPerf(treetuneClusterrpart)

'Accuracy = 0.9221482'

treeClusterrpart <- rpart(ClusterPortafoglio ~ . - Match - ClusterProfili, data=dataClusterTrain, cp=0.007)
rpart.plot(treeClusterrpart, type = 4, extra=101)
prp(treeClusterrpart, extra=101,  varlen = -10)

plot(varImp(treetuneClusterrpart))


# Feature Selection
treetuneClusterrpart$finalModel$variable.importance
# secondo rpart sarebbero tutte da prendere, solo ClientKnowledgeExperience ha un valore basso.


# RandomForest

set.seed(7)
RandomForestCluster <- randomForest(formula = ClusterPortafoglio ~ . - Match - ClusterProfili, data = dataset_finale_training)
RandomForestCluster # n tree = 500, mtry=radice del numero delle variabili
confusionMatrix(RandomForestCluster$predicted, dataset_finale_training$ClusterPortafoglio)

performanceRF <- cbind(t(confusionMatrix(RandomForestCluster$predicted, dataset_finale_training$ClusterPortafoglio)$'overall'[1:2]), 
                       method = 'Random Forest')
colnames(performanceRF) <- c('TrainAccuracy', 'TrainKappa', 'method')

'Accuracy = 0.9675'

RandomForestCluster$importance

# Reti neurali

# Inheritance Index da togliere perchè collineare con altre variabili

library(nnet)
library(NeuralNetTools) # per plot rete neurale

set.seed(7)
metric <- "Accuracy"
tunegrid <- expand.grid(size=c(1:7), decay = c(0.001, 0.01, 0.05 , .1, .3))
nnetCluster <- train(dataset_finale_training[,-c(8:11)], dataset_finale_training$ClusterPortafoglio, method = "nnet",
                     preProcess = c("scale", 'center'), tuneLength = 10, metric=metric, tuneGrid=tunegrid, trace = FALSE, maxit = 300)

nnetCluster # migliore con 6 hidden units e 0.1 di decay
getTrainPerf(nnetCluster)
plot(nnetCluster)
plotnet(nnetCluster$finalModel, varlen = -10)

'Accuracy = 0.9731533'

varImp(nnetCluster)

# NaiveBayes

findCorrelation(cor(dataset_finale_training[-c(9:11)]), cutoff = 0.6, names = TRUE) # individua la variabili da togliere in base alla correlazione

# Anche in questo caso tolgo Inheritance Index siccome rindondante

set.seed(7)
metric <- "Accuracy"
NBCluster <- train(ClusterPortafoglio ~ . - Match - ClusterProfili - InheritanceIndex, data = dataClusterTrain, method="nb", metric=metric)
NBCluster # use kernel = FALSE
getTrainPerf(NBCluster)

'Accuracy = 0.9551633'


# SVM

set.seed(7)
metric <- "Accuracy"
SVMCluster <- train(ClusterPortafoglio ~ . - Match - ClusterProfili - InheritanceIndex, data = dataClusterTrain, method="svmRadial",
                    metric=metric, preProcess = c("scale","center"))
SVMCluster # sigma = 0.1604924 and C = 1
getTrainPerf(SVMCluster)

'Accuracy = 0.9670034'

# Bagged CART

set.seed(7)
metric <- "Accuracy"
BaggedTreeCluster <- train(ClusterPortafoglio ~ . - Match - ClusterProfili, data = dataClusterTrain, method="treebag", metric=metric)
BaggedTreeCluster 
getTrainPerf(BaggedTreeCluster)

'Accuracy = 0.9465458'

# Stochastic Gradient Boosting (Generalized Boosted Modeling)

library(gbm)
set.seed(7)
metric <- "Accuracy"
GradientBoostingCluster <- train(ClusterPortafoglio ~ . - Match - ClusterProfili, data = dataClusterTrain, method="gbm", metric=metric)
GradientBoostingCluster # n.trees = 150, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10
getTrainPerf(GradientBoostingCluster)

'Accuracy = 0.9671624'


# Tabella di Accuracy

Accuracy <- rbind(getTrainPerf(GradientBoostingCluster), getTrainPerf(BaggedTreeCluster), getTrainPerf(SVMCluster), getTrainPerf(NBCluster),
                  getTrainPerf(nnetCluster), getTrainPerf(treetuneClusterrpart), performanceRF)
Accuracy <- Accuracy %>% arrange(desc(Accuracy$TrainAccuracy))
Accuracy$TrainKappa <- NULL
colnames(Accuracy) <- c('Accuracy', 'Model')

'Miglior modello è la rete neurale'

# Predict on Test set the best model

dataset_finale_test$Prediction <- predict(nnetCluster, dataset_finale_test, 'raw')
table(dataset_finale_test$ClusterProfili, dataset_finale_test$Prediction, dnn = c('ClusterProfili', 'ClusterPortafogliPredetti'))
(1383+1118+920)/3708

'Accuracy Test = 0.9225998'

#####
# REGRESSIONE SU RISK PROPENSION E CLIENT INVESTMENT HORIZON
#####

### REGRESSIONE SU VARIABILI DATASET

dati_clienti_NOPortfolio <- dati_clienti[, -c(1, 3, 5, 19:24)]

status_NoPortfolio <- df_status(dati_clienti_NOPortfolio)
QuantNoPortfolio <- filter(status_NoPortfolio,  type %in% c("numeric", "integer")) %>% .$variable
dati_clienti_NOPortfolioQ <- select(dati_clienti_NOPortfolio, one_of(QuantNoPortfolio))

cor_NoPortfolio <- cor(dati_clienti_NOPortfolioQ)
b_NoPortfolio <- subset(melt(cor_NoPortfolio), value > .6 | value < -0.6) %>% filter(value != 1)
corHigh_NoPortfolio <- b_NoPortfolio[!duplicated(b_NoPortfolio$value), ]

# Age e Inheritance Index sono correlate con molte variabili, da togliere per regressione

## Regressione su Risk Propension

corRisk <- cor(dati_clientiQ$RiskPropension, dati_clientiQ)
corRisk[, which(corRisk< -0.5 | corRisk > .5 & corRisk !=1)] # Risk Propension è correlata con Age, PensionNeed, InheritanceIndex

modelRisk <- lm(data = dati_clienti_NOPortfolio, formula = RiskPropension ~ .)
summary(modelRisk)
vif(modelRisk) # multicollineare Inheritance Index e Age
modelRisk2 <- lm(data = dati_clienti_NOPortfolio, formula = RiskPropension ~ . - Age - InheritanceIndex) 
# tolgo Age e Inheritance Index perchè il modello se no rimane eteroschedastico
summary(modelRisk2)
vif(modelRisk2) # no multicollinearità

stepwiseRisk <- step(modelRisk2, direction = 'both')
summary(stepwiseRisk)

gvlma(stepwiseRisk) # eteroschedasticità
dwtest(stepwiseRisk) # incorrelati

plot(stepwiseRisk, which = 4, id.n = 3)  #plot cook's distance: presenza di osservazioni influenti
abline(a=0.001,b=0,col='red') #sarebbe meglio eliminare almeno le osservazioni sopra questa soglia
aug1=augment(stepwiseRisk) %>%   mutate(index = 1:n())
dcook=aug1$.cooksd
dcook=as.data.frame(dcook)
which(dcook>0.001)

modelRisk_cleaned <- lm(data = dati_clienti_NOPortfolio[-(which(dcook>0.001)),], formula = stepwiseRisk$call$formula)
summary(modelRisk_cleaned)
stepwiseRisk2 <- step(modelRisk_cleaned)
summary(stepwiseRisk2)
gvlma(stepwiseRisk2) # omoschedastico

par(mfrow = c(2, 2))
plot(stepwiseRisk2)
par(mfrow = c(1, 1))

## Regressione ClientInvestmentHorizon

modelHorizon <- lm(data = dati_clienti_NOPortfolio, formula = ClientInvestmentHorizon ~ .)
summary(modelHorizon)
vif(modelHorizon) # Age multicollineare

modelHorizon2 <- lm(data = dati_clienti_NOPortfolio, formula = ClientInvestmentHorizon ~ . - Age)
summary(modelHorizon2)
vif(modelHorizon2) # No multicollinearità

stepwiseHorizon <- step(modelHorizon2, direction = 'both')
summary(stepwiseHorizon)

gvlma(stepwiseHorizon) # omoschedastico
dwtest(modelHorizon2) # incorrelati

plot(modelHorizon2, which = 4, id.n = 3)  #plot cook's distance: presenza di osservazioni influenti
abline(a=0.001,b=0,col='red') #sarebbe meglio eliminare almeno le osservazioni sopra questa soglia
aug1=augment(modelHorizon2) %>%   mutate(index = 1:n())
dcook=aug1$.cooksd
dcook=as.data.frame(dcook)
which(dcook>0.001)

modelHorizon_cleaned <- lm(data = dati_clienti_NOPortfolio[-(which(dcook>0.001)),], formula = stepwiseHorizon$call$formula)
summary(modelHorizon_cleaned)
stepwiseHorizon2 <- step(modelHorizon_cleaned)
summary(stepwiseHorizon2)

gvlma(stepwiseHorizon2) # omoschedastico

par(mfrow = c(2, 2))
plot(stepwiseHorizon2)
par(mfrow = c(1, 1))

'Impossibile regressione facciamo alberi'

#####
# LOGISTICA SU INCOME HIGH LOW E NO TRUST IN BANKS
#####
### Logistic Regression su alcune variabili binarie (Income High Low, No trust in banks)

dati_clienti_good <- dati_clienti[, -c(1, 19)] # tolgo provincia client ID

# Preprocessing per logistica
# controllo correlazione tra variabili
cor <- cor(dati_clientiQ)
b <- subset(melt(cor), value > .5 | value < -0.5) %>% filter(value != 1)
corHigh <- b[!duplicated(b$value), ]
# togliamo Age, Equity investments, Inheritance index, Cash


## Logistica su IncomeHighLow
modelInc <- glm(data = dati_clienti_good, formula = IncomeHighLow ~ . - Age - InheritanceIndex - EquityInvestments - Cash, family = binomial)
summary(modelInc)
stepwiseInc <- step(modelInc, direction = 'both')
summary(stepwiseInc)

"la logistica non si basa sulle ipotesi classiche del modello di reg. lineare; per la diagnostica bisognerebbe fare analisi
multicollinearità e valori influenti."

#verifico multicollinearità
vif(stepwiseInc)# tutti gli indici VIF sono sotto il valore soglia 5, non c'è collinearità (siamo a posto)
#mancherebbe analisi valori influenti

#bisogna verificare la presenza di autocorrelazione in un logistico?
dwtest(stepwiseInc) # non c'è autocorrelazione

plot(stepwiseInc, which = 4, id.n = 3)  #plot cook's distance: presenza di osservazioni influenti
soglia_CookD=(4/dim(dati_clienti_good)[1])
abline(a=soglia_CookD,b=0,col='red')
abline(a=0.02,b=0,col='red') #sarebbe meglio eliminare almeno le osservazioni sopra questa soglia
aug1=augment(stepwiseInc) %>%   mutate(index = 1:n())
dcook=aug1$.cooksd
dcook=as.data.frame(dcook)
which(dcook>0.02)# indici di riga punti influenti

#calcolo Nagelkerke Rsquared
modnull_inc=glm(data = dati_clienti_good, formula = IncomeHighLow ~ -.,family='binomial')
lr.stat <- lrtest(modnull_inc, stepwiseInc)
(1-exp(-(as.numeric(lr.stat$stats[1]))/5000))/(1-exp(2*as.numeric(logLik(modnull_inc)/5000)))#nagelkerke Rsquared

modelInc_cleaned <- glm(data = dati_clienti_good[-(which(dcook>0.02)),], formula = stepwiseInc$formula, family = binomial)
summary(modelInc_cleaned)#AIC migliora da 1257 a 1216
lr.stat <- lrtest(modnull_inc, modelInc_cleaned)
(1-exp(-(as.numeric(lr.stat$stats[1]))/5000))/(1-exp(2*as.numeric(logLik(modnull_inc)/5000)))
#rquadro migliora di poco

# Odds Ratio
sapply(stepwiseInc$coefficients[2:length(stepwiseInc$coefficients)], function(x) exp(x))
sapply(modelInc_cleaned$coefficients[2:length(modelInc_cleaned$coefficients)], function(x) exp(x))

' Di conseguenza le variabili significative per il reddito dichiarato (IncomeHighLow) sono PortfolioHorizon, ClientPotentialIndex, Sex, AuM,
IncomeNeed, LongTermCareNeed, PensionNeed, PanicMood (nel nuovo modello pulito non significativo), NoTrustInBanks,
OtherInvestments (nel nuovo modello pulito non significativo).

'


## regressione logistica + stepwise, con Notrustinbanks come target

modelTrust <- glm(data = dati_clienti_good, formula = NoTrustInBanks ~ . - Age - InheritanceIndex - EquityInvestments - Cash, family = binomial)
summary(modelTrust)
stepwiseTrust <- step(modelTrust, direction = 'both')
summary(stepwiseTrust)

vif(stepwiseTrust) # no collinearità
dwtest(stepwiseTrust) # no autocorrelazione

plot(stepwiseTrust, which = 4, id.n = 3)  #plot cook's distance
abline(a=0.02,b=0,col='red')
soglia_CookD=(4/dim(dati_clienti_good)[1])
aug1=augment(stepwiseTrust) %>%   mutate(index = 1:n())
dcook=aug1$.cooksd
dcook=as.data.frame(dcook)
which(dcook>0.02)

modnull_inc=glm(data = dati_clienti_good, formula = NoTrustInBanks ~ -.,family='binomial')
lr.stat <- lrtest(modnull_inc, modelTrust)
(1-exp(-(as.numeric(lr.stat$stats[1]))/5000))/(1-exp(2*as.numeric(logLik(modnull_inc)/5000)))#R2 basso

modelTrust_cleaned <- glm(data = dati_clienti_good[-(which(dcook>0.02)),], formula = stepwiseTrust$formula, family = binomial)
summary(modelTrust_cleaned)#AIC migliora da 3136 a 3087
lr.stat <- lrtest(modnull_inc, modelTrust_cleaned)
(1-exp(-(as.numeric(lr.stat$stats[1]))/5000))/(1-exp(2*as.numeric(logLik(modnull_inc)/5000)))

# Odds Ratio
sapply(modelTrust_cleaned$coefficients[2:length(modelTrust_cleaned$coefficients)], function(x) exp(x))


'
Le variabili significative per NoTrustInBanks sono PortfolioRisk, PortfolioHorizon, ClientKnowledgeExperience, ClientPotentialIndex, IncomeHighLow,
IncomeNeed, BondInvestments, MoneyMarketInvestments.'




#####
# ALBERO SU PANIC MOOD
#####


dataPM <- dati_clienti_NOPortfolio
dataPM$PanicMood <- make.names(dataPM$PanicMood)
dataPM$PanicMood <- as.factor(dataPM$PanicMood)
str(dataPM)
table(dataPM$PanicMood) #- sbilanciato verso valore negativo

## Setto solo CP siccome minsplit potrebbe essere inutile visto che non arriva mai ad un nodo più piccolo di 35
set.seed(7)
control <- trainControl(method="cv", number=10, summaryFunction = twoClassSummary, classProbs = TRUE)
tunegrid <- expand.grid(cp=seq(0.01, 0.15, 0.002))
treetunePMrpart <- train(PanicMood ~ ., data=dataPM, method='rpart', tuneGrid=tunegrid, trControl=control, metric='Spec')
treetunePMrpart

treePMrpart <- rpart(PanicMood ~ ., dataPM, method = 'class', cp = 0.044)
rpart.plot(treePMrpart, type = 4, extra=101)
prp(treePMrpart, extra=101, varlen = -10)

plot(varImp(treetunePMrpart))

'Le variabili importanti sono ProtectionNeed, ClientInvestmentHorizon, IncomeNeed, Age, PensionNeed'

## Setto cp e minsplit con CustomTree (si appoggia a RandomForest)
set.seed(7)
control <- trainControl(method="cv", number=10, summaryFunction = twoClassSummary, classProbs = TRUE)
tunegrid <- expand.grid(cp=seq(0.005, 0.15, 0.002), minsplit=seq(10, 25, 5) )
treetunePM <- train(PanicMood ~ ., data=dataPM, method=customTree, tuneGrid=tunegrid, trControl=control, metric='Spec')
# uso la specificity essendo un imbalanced problem verso valori negativi
treetunePM

# usiamo rpart solo per plottare
treePM <- rpart(PanicMood ~ ., dataPM, method = 'class', cp = 0.065, minsplit=25)
rpart.plot(treePM, type = 4, extra=101)

# usiamo albero caret per vedere le variabili importanti
plot(varImp(treetunePM))
'le variabili più importanti sono ClientInvestmentHorizon, ProtectionNeed, ClientKnoledgeExperience e PortfolioRisk'

# in questo caso togliendo le variabili di portafoglio esce fuori lo stesso albero

"# ora riproviamo a fare l'albero solo con le variabili del questionario e non del portafoglio
dataPM2 <- dataPM[, -c(1, 3, 18:22)]
set.seed(7)
control <- trainControl(method='cv', number=10, summaryFunction = twoClassSummary, classProbs = TRUE)
tunegrid <- expand.grid(cp=seq(0.005, 0.15, 0.002), minsplit=seq(10, 25, 5) )
treetunePM2 <- train(PanicMood ~ ., data=dataPM2, method=customTree, tuneGrid=tunegrid, trControl=control, metric='Spec')
# uso la specificity essendo un imbalanced problem
treetunePM2

# usiamo rpart solo per plottare
treePM2 <- rpart(PanicMood ~ ., dataPM2, method = 'class', cp = 0.02, minsplit=10)
rpart.plot(treePM2, type = 4, extra=101)

# usiamo albero caret per vedere le variabili importanti
plot(varImp(treetunePM2))
"

