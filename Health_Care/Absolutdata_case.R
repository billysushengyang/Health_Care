rm(list = ls(all = TRUE))
setwd("C:/Users/chang_000/Dropbox/Pinsker/CV_Resume/2016JobSearch/Absolutdata/")

#
# define functions
#
f_usePackage <- function(p) {
    if (!is.element(p, installed.packages()[,1])) {
        install.packages(p, dep = TRUE);
    }
    require(p, character.only = TRUE);
}


#
# load library
#
f_usePackage("data.table")
f_usePackage("magrittr")
f_usePackage("dplyr")
f_usePackage("gmodels")
f_usePackage("caret")
f_usePackage("xgboost")
f_usePackage("pROC")
f_usePackage("e1071")


#############################################################
# Load data and summarize it
#
df <- fread("Absolutdata_case.csv")

# remove the duplicates if there are any in the data set
df <- df[!duplicated(df), ]

#############################################################
# Data Transformation
df$Age_binned <- cut(df$Age, 
                     breaks = c(0, seq(20, 70, 5), Inf), 
                     labels = c("<=20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61-65", "66-70", ">70"))

#############################################################
# Data Preparation
# 
## log-transformation of Level_of_Hemoglobin
df$log_Level_of_Hemoglobin <- log(df$Level_of_Hemoglobin, base = 10)

# box plot
boxplot(log_Level_of_Hemoglobin ~ Blood_Pressure_Abnormality,data = df, xlab = "Blood_Pressure_Abnormality", ylab = "log_Level_of_Hemoglobin")
t.test(subset(df$log_Level_of_Hemoglobin, df$Blood_Pressure_Abnormality == 0), subset(df$log_Level_of_Hemoglobin, df$Blood_Pressure_Abnormality == 1))
# histogram
x <- df$log_Level_of_Hemoglobin
h <- hist(x, breaks = 10, col = "light blue", xlab = "Level_of_Hemoglobin", main = "Histogram with Normal Curve") 
xfit <- seq(min(x), max(x), length = 40) 
yfit <- dnorm(xfit, mean = mean(x), sd = sd(x)) 
yfit <- yfit * diff(h$mids[1:2]) * length(x) 
lines(xfit, yfit, col = "blue", lwd = 2)

# imputation of Genetic_Pedigree_Coefficient
for (i in 1:nrow(df)) {
    if (is.na(df$Genetic_Pedigree_Coefficient[i])) {
        df$Genetic_Pedigree_Coefficient[i] = median(df$Genetic_Pedigree_Coefficient[df$Sex==df$Sex[i] & df$Age_binned==df$Age_binned[i] & df$Smoking==df$Smoking[i]], na.rm=TRUE) 
    } 
}

# encoding NA as Missing" in Pregnancy 
df$Pregnancy[is.na(df$Pregnancy)] <- "Missing"

# imputation of alcohol_consumption_per_day
for (i in 1:nrow(df)) {
    if (is.na(df$alcohol_consumption_per_day[i])) {
        df$alcohol_consumption_per_day[i] = median(df$alcohol_consumption_per_day[df$Sex==df$Sex[i] & df$Smoking==df$Smoking[i] & df$Level_of_Stress==df$Level_of_Stress[i]], na.rm=TRUE) 
    } 
}

# check NAs by column
apply(is.na(df), 2, sum)


# Define data types of variables 
# is factors:
is_factor_cols <- c("Blood_Pressure_Abnormality", "Sex", "Pregnancy", "Level_of_Stress", "Chronic_kidney_disease", "Adrenal_and_thyroid_disorders", "Smoking")
lapply(is_factor_cols, function(x) {
    e = substitute(X := as.factor(X), list(X = as.symbol(x)))
    df[, eval(e)]})

#############################################################
# Modeling
#

# define target variable and check if the target variable is balanced or imbalanced
target_variable = c("Blood_Pressure_Abnormality")
table(df[[target_variable]]) #balanced!

# remove patient ID from data frame
df$Patient_Number <- NULL

# remove Level_of_Hemoglobin and Age_binned from data frame
df$Level_of_Hemoglobin <- NULL
df$Age_binned          <- NULL

# set up of K-fold CV for validation
K <- 5
block <- sample(1:K, nrow(df), replace=TRUE)

output <- data.frame(xgb = rep(0, K), rf = rep(0, K), svm = rep(0, K))
for (i in 1:K) {
    train <- df[block != i,] 
    test  <- df[block == i,]
    
    # Normalization of numeric vars in training set
    numeric_cols <- vapply(df, is.numeric, logical(1))
    preObj <- preProcess(train[numeric_cols], method = c("center", "scale"))
    train  %<>% predict(preObj, .)
    train_x <- dplyr::select(train, -c(Blood_Pressure_Abnormality))
    train_y <- train[[target_variable]]
    
    # Normalization of numeric vars in test set
    test %<>% predict(preObj, .)
    test_x <- dplyr::select(test, -Blood_Pressure_Abnormality)
    test_y <- test[[target_variable]]
     
    
    ##########
    # XGB
    ##########
    options_na_action_restore <- options(na.action = "na.pass")
    on.exit(options(options_na_action_restore), add = TRUE)
    
    # xgb training set
    xgb_train        <- model.matrix(Blood_Pressure_Abnormality ~ ., data = train)[, -1]
    xgb_train_label  <- as.numeric(as.character(train[[target_variable]]))
    xgb_train_dmatrix <- xgb.DMatrix(data = xgb_train, label = xgb_train_label, missing = NA)
    
    # xgb test set
    xgb_test         <- model.matrix(Blood_Pressure_Abnormality ~ ., data = test)[, -1]
    xgb_test_label   <- as.numeric(as.character(test[[target_variable]]))
    xgb_test_dmatrix <- xgb.DMatrix(data = xgb_test, label = xgb_test_label, missing = NA)
    
    # XGB parameters
    eta     <- 0.005
    nrounds <- 100 #5000
    searchGridSubCol <- expand.grid(subsample          = seq(0.7, 0.8, 0.1) 
                                    , colsample_bytree = seq(0.5, 0.7, 0.2)
                                    , max_depth        = c(5, 8)
                                    , lambda           = 0.5
                                    , alpha            = 0.25
                                    , gamma            = 0.01)

    tune_Hyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
        xgboostModelCV <- xgb.cv(data               = xgb_train_dmatrix
                                 , objective        = "binary:logistic"
                                 , metrics          = "auc"
                                 , nrounds          = nrounds
                                 , eta              = eta
                                 , nfold            = 5
                                 , max_depth        = parameterList[["max_depth"]]                               
                                 , subsample        = parameterList[["subsample"]]
                                 , colsample_bytree = parameterList[["colsample_bytree"]]
                                 , lambda           = parameterList[["lambda"]]
                                 , alpha            = parameterList[["alpha"]]
                                 , gamma            = parameterList[["gamma"]]
                                 , missing          = NA
                                 , nthread          = parallel::detectCores())
        xvalidationScores <- as.data.frame(xgboostModelCV)
        return(tail(xvalidationScores$test.auc.mean, 1))
    })
    searchGridSubCol$auc <- tune_Hyperparameters
    searchGridSubCol <- searchGridSubCol[order(searchGridSubCol$auc, decreasing = TRUE),]
    
    # final XGB model
    xgb <- xgboost(data                = xgb_train_dmatrix 
                   , objective         = "binary:logistic"
                   , eval_metrics      = "auc"
                   , nrounds           = 2000
                   , eta               = eta
                   , max_depth         = searchGridSubCol$max_depth[1]                     
                   , subsample         = searchGridSubCol$subsample[1]  
                   , colsample_bytree  = searchGridSubCol$colsample_bytree[1]
                   , lambda            = searchGridSubCol$lambda[1]
                   , alpha             = searchGridSubCol$alpha[1]
                   , gamma             = searchGridSubCol$gamma[1]
                   , early.stop.round  = 20
                   , missing           = NA
                   , seed              = 1234
                   , nthread           = parallel::detectCores()
    )

    # predict on test
    xgb_test_pred <- predict(xgb, xgb_test_dmatrix)
    output[i, 1]  <- auc(xgb_test_label, xgb_test_pred)
    
    
    ###############  
    # Random Forest
    ############### 
    # Tune RF parameters
    folds <- sample(1:5, nrow(train), replace = TRUE)
   
    mtry_list  <- seq(3, 15, 3)
    ntree_list <- 25
    
    cv_output_rf <- vector()
    for (p in 1:length(mtry_list)) {
        for (q in 1:length(ntree_list)) {
            auc_list   <- vector()
            for (j in 1:5) {
                mod = randomForest(x           = data.frame(train_x)[folds != j,]
                                   , y         = train_y[folds != j]
                                   , mtry      = mtry_list[p]
                                   , ntree     = ntree_list[q]
                                   )
                pred     <- predict(mod, data.frame(train_x)[folds == j,], type="prob")[,2]
                auc      <- auc(train_y[folds == j], pred)
                auc_list <- c(auc_list, auc)
            }
            auc <- mean(auc_list)
            a <- c(mtry = mtry_list[p], ntree = ntree_list[q], auc = auc)
            cv_output_rf <- rbind(cv_output_rf, a)
        }
    }
    cv_output_rf <- data.frame(cv_output_rf)
    cv_output_rf <- cv_output_rf[order(cv_output_rf$auc, decreasing = TRUE),]
    
    # final rf model
    rf <- randomForest(x           = data.frame(train_x)[, !colnames(train_x) %in% c("alcohol_consumption_per_day", "Pregnancy", "Genetic_Pedigree_Coefficient")]
                       , y         = train_y
                       , mtry      = cv_output_rf[1, c("mtry")]
                       , ntree     = cv_output_rf[1, c("ntree")]
    )
    
    # predict on test
    rf_test_pred <- predict(rf, data.frame(test_x)[, !colnames(test_x) %in% c("alcohol_consumption_per_day", "Pregnancy", "Genetic_Pedigree_Coefficient")], type="prob")[,2]
    output[i, 2] <- auc(as.numeric(as.character(test_y)), rf_test_pred)

    
    ###############  
    # SVM
    ###############  
    options_na_action_restore <- options(na.action = "na.pass")
    on.exit(options(options_na_action_restore), add = TRUE)
    
    # svm training set
    svm_train         <- model.matrix( ~ ., data = train)[, -1]
    svm_train_label   <- as.numeric(as.character(train[[target_variable]]))

    # svm test set
    svm_test         <- model.matrix( ~ ., data = test)[, -1]
    svm_test_label   <- as.numeric(as.character(test[[target_variable]]))

    # SVM final model
    svm <- svm(Blood_Pressure_Abnormality1 ~ .
               , data        = svm_train
               , kernel      = "radial"
               , cost        = 5
               , gamma       = 0.1
               , probability = TRUE
               , type        = "C"
               )

    # predict on test
    pred.svm      <- predict(svm, svm_test, probability = TRUE)
    svm_test_pred <- attr(pred.svm, "probabilities")[,2]
    output[i, 3]  <- auc(test_y, svm_test_pred)
}

output
# xgb        rf        svm
#1 0.9388987 0.8283128 0.9133518
#2 0.9331225 0.8282894 0.8903975
#3 0.9497494 0.8307161 0.9071055
#4 0.9281408 0.8111034 0.8886837
#5 0.9490415 0.8539378 0.9171503

##################################################
# FINAL MODEL:
# Use FULL data set to train the model as final
##################################################
options_na_action_restore <- options(na.action = "na.pass")
on.exit(options(options_na_action_restore), add = TRUE)

# Normalization 
numeric_cols <- vapply(df, is.numeric, logical(1))
preObj       <- preProcess(df[numeric_cols], method = c("center", "scale"))
df           %<>% predict(preObj, .)
df_x         <- dplyr::select(df, -c(Blood_Pressure_Abnormality))
df_y         <- df[[target_variable]]

# xgb full data set
xgb_df         <- model.matrix(Blood_Pressure_Abnormality ~ ., data = df)[, -1]
xgb_df_label   <- as.numeric(as.character(df_y))
xgb_df_dmatrix <- xgb.DMatrix(data = xgb_df, label = xgb_df_label, missing = NA)

# tune XGB parameters of final model
eta     <- 0.005
nrounds <- 100 #5000
searchGridSubCol <- expand.grid(subsample          = seq(0.7, 0.8, 0.1) 
                                , colsample_bytree = seq(0.5, 0.7, 0.2)
                                , max_depth        = c(5, 8)
                                , lambda           = 0.5
                                , alpha            = 0.25
                                , gamma            = 0.01)

tune_Hyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
    xgboostModelCV <- xgb.cv(data               = xgb_df_dmatrix
                             , objective        = "binary:logistic"
                             , metrics          = "auc"
                             , nrounds          = nrounds
                             , eta              = eta
                             , nfold            = 5
                             , max_depth        = parameterList[["max_depth"]]                               
                             , subsample        = parameterList[["subsample"]]
                             , colsample_bytree = parameterList[["colsample_bytree"]]
                             , lambda           = parameterList[["lambda"]]
                             , alpha            = parameterList[["alpha"]]
                             , gamma            = parameterList[["gamma"]]
                             , missing          = NA
                             , nthread          = parallel::detectCores())
    xvalidationScores <- as.data.frame(xgboostModelCV)
    return(tail(xvalidationScores$test.auc.mean, 1))
})
searchGridSubCol$auc <- tune_Hyperparameters
searchGridSubCol <- searchGridSubCol[order(searchGridSubCol$auc, decreasing = TRUE),]

# final XGB model
xgb <- xgboost(data                = xgb_df_dmatrix 
               , objective         = "binary:logistic"
               , eval_metrics      = "auc"
               , nrounds           = 8000
               , eta               = eta
               , max_depth         = searchGridSubCol$max_depth[1]                     
               , subsample         = searchGridSubCol$subsample[1]  
               , colsample_bytree  = searchGridSubCol$colsample_bytree[1]
               , lambda            = searchGridSubCol$lambda[1]
               , alpha             = searchGridSubCol$alpha[1]
               , gamma             = searchGridSubCol$gamma[1]
               , early.stop.round  = 100
               , missing           = NA
               , seed              = 1234
               , nthread           = parallel::detectCores()
)

# Variable Importance
feature_names  <- colnames(df)[-grep("Blood_Pressure_Abnormality", colnames(df))]
important_vars <- xgb.importance(feature_names = feature_names, model = xgb)

# Variable Importance Plot
xgb.plot.importance(important_vars)

# save objects
model_container <- vector("list")

model_container$train          <- df
model_container$preProcess_obj <- preObj
model_container$model          <- xgb

save(model_container, file = "model_container.rda")

################################################################

load(file = "model_container.rda")
df     <- model_container$train
xgb    <- model_container$model
preObj <- model_container$preProcess_obj

newdata <- fread("xxx.csv")

# define target variable 
target_variable = c("Blood_Pressure_Abnormality")

## Data Manipulation:
# binning Age
newdata$Age_binned <- cut(newdata$Age
                          , breaks = c(0, seq(20, 70, 5), Inf)
                          , labels = c("<=20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61-65", "66-70", ">70"))


# log-transformation of Level_of_Hemoglobin
df$log_Level_of_Hemoglobin <- log(df$Level_of_Hemoglobin, base = 10)

# imputation of Genetic_Pedigree_Coefficient
for (i in 1:nrow(newdata)) {
    if (is.na(newdata$Genetic_Pedigree_Coefficient[i])) {
        newdata$Genetic_Pedigree_Coefficient[i] = median(df$Genetic_Pedigree_Coefficient[df$Sex==df$Sex[i] & df$Age_binned==df$Age_binned[i] & df$Smoking==df$Smoking[i]], na.rm=TRUE) 
    } 
}

# encoding NA as Missing" in Pregnancy 
newdata$Pregnancy[is.na(newdata$Pregnancy)] <- "Missing"

# imputation of alcohol_consumption_per_day
for (i in 1:nrow(newdata)) {
    if (is.na(newdata$alcohol_consumption_per_day[i])) {
        newdata$alcohol_consumption_per_day[i] = median(df$alcohol_consumption_per_day[df$Sex==df$Sex[i] & df$Smoking==df$Smoking[i] & df$Level_of_Stress==df$Level_of_Stress[i]], na.rm=TRUE) 
    } 
}

# Define data types of variables 
# is factors:
is_factor_cols <- c("Blood_Pressure_Abnormality", "Sex", "Pregnancy", "Level_of_Stress", "Chronic_kidney_disease", "Adrenal_and_thyroid_disorders", "Smoking")
lapply(is_factor_cols, function(x) {
    e = substitute(X := as.factor(X), list(X = as.symbol(x)))
    newdata[, eval(e)]})

# remove variables from data frame
newdata$Patient_Number      <- NULL
newdata$Level_of_Hemoglobin <- NULL
newdata$Age_binned          <- NULL

# Normalization of numeric vars in test set
newdata %<>% predict(preObj, .)
newdata_x <- dplyr::select(newdata, -Blood_Pressure_Abnormality)
newdata_y <- newdata[[target_variable]]

## Data Preparation:
# build xgb dmatrix for newdata
xgb_newdata         <- model.matrix(Blood_Pressure_Abnormality ~ ., data = newdata)[, -1]
xgb_newdata_label   <- as.numeric(as.character(newdata[[target_variable]]))
xgb_newdata_dmatrix <- xgb.DMatrix(data = xgb_newdata, label = xgb_newdata_label, missing = NA)

# Scoring:
xgb_newdata_pred <- predict(xgb, xgb_newdata_dmatrix)
auc(xgb_newdata_label, xgb_newdata_pred)

#############################################################
# Exploratory Analysis 
#

# Boxplots:
# boxplot of Level_of_Hemoglobin by Blood_Pressure_Abnormality
boxplot(Level_of_Hemoglobin ~ Blood_Pressure_Abnormality,data = df, xlab = "Blood_Pressure_Abnormality", ylab = "Level_of_Hemoglobin")
t.test(subset(df$Level_of_Hemoglobin, df$Blood_Pressure_Abnormality == 0), subset(df$Level_of_Hemoglobin, df$Blood_Pressure_Abnormality == 1))

x <- df$Level_of_Hemoglobin
h <- hist(x, breaks = 10, col = "light blue", xlab = "Level_of_Hemoglobin", main = "Histogram with Normal Curve") 
xfit <- seq(min(x), max(x), length = 40) 
yfit <- dnorm(xfit, mean = mean(x), sd = sd(x)) 
yfit <- yfit * diff(h$mids[1:2]) * length(x) 
lines(xfit, yfit, col = "blue", lwd = 2)


# boxplot of Genetic_Pedigree_Coefficient by Blood_Pressure_Abnormality
boxplot(Genetic_Pedigree_Coefficient ~ Blood_Pressure_Abnormality,data = df, xlab = "Blood_Pressure_Abnormality", ylab = "Genetic_Pedigree_Coefficient")
t.test(subset(df$Genetic_Pedigree_Coefficient, df$Blood_Pressure_Abnormality == 0), subset(df$Genetic_Pedigree_Coefficient, df$Blood_Pressure_Abnormality == 1))

x <- df$Genetic_Pedigree_Coefficient
h <- hist(x, breaks = 10, col = "light blue", xlab = "Level_of_Hemoglobin", main = "Histogram with Normal Curve") 
xfit <- seq(min(x), max(x), length = 40) 
yfit <- dnorm(xfit, mean = mean(x), sd = sd(x)) 
yfit <- yfit * diff(h$mids[1:2]) * length(x) 
lines(xfit, yfit, col = "blue", lwd = 2)


# boxplot of Age by Blood_Pressure_Abnormality
boxplot(Age ~ Blood_Pressure_Abnormality,data = df, xlab = "Blood_Pressure_Abnormality", ylab = "Age")
t.test(subset(df$Age, df$Blood_Pressure_Abnormality == 0), subset(df$Age, df$Blood_Pressure_Abnormality == 1))

x <- df$Age
h <- hist(x, breaks = 10, col = "light blue", xlab = "Age", main = "Histogram with Normal Curve") 
xfit <- seq(min(x), max(x), length = 40) 
yfit <- dnorm(xfit, mean = mean(x), sd = sd(x)) 
yfit <- yfit * diff(h$mids[1:2]) * length(x) 
lines(xfit, yfit, col = "blue", lwd = 2)


# boxplot of BMI by Blood_Pressure_Abnormality
boxplot(BMI ~ Blood_Pressure_Abnormality,data = df, xlab = "Blood_Pressure_Abnormality", ylab = "BMI")
t.test(subset(df$BMI, df$Blood_Pressure_Abnormality==0), subset(df$BMI, df$Blood_Pressure_Abnormality==1))

x <- df$BMI
h <- hist(x, breaks = 10, col = "light blue", xlab = "BMI", main = "Histogram with Normal Curve") 
xfit <- seq(min(x), max(x), length = 40) 
yfit <- dnorm(xfit, mean = mean(x), sd = sd(x)) 
yfit <- yfit * diff(h$mids[1:2]) * length(x) 
lines(xfit, yfit, col = "blue", lwd = 2)


# boxplot of Physical_activity by Blood_Pressure_Abnormality
boxplot(Physical_activity ~ Blood_Pressure_Abnormality,data = df, xlab = "Blood_Pressure_Abnormality", ylab = "Physical_activity")
t.test(subset(df$Physical_activity, df$Blood_Pressure_Abnormality==0), subset(df$Physical_activity, df$Blood_Pressure_Abnormality==1))

x <- df$Physical_activity
h <- hist(x, breaks = 10, col = "light blue", xlab = "Physical_activity", main = "Histogram with Normal Curve") 
xfit <- seq(min(x), max(x), length = 40) 
yfit <- dnorm(xfit, mean = mean(x), sd = sd(x)) 
yfit <- yfit * diff(h$mids[1:2]) * length(x) 
lines(xfit, yfit, col = "blue", lwd = 2)


# boxplot of salt_content_in_the_diet by Blood_Pressure_Abnormality
boxplot(salt_content_in_the_diet ~ Blood_Pressure_Abnormality,data = df, xlab = "Blood_Pressure_Abnormality", ylab = "BMI")
t.test(subset(df$salt_content_in_the_diet, df$Blood_Pressure_Abnormality==0), subset(df$salt_content_in_the_diet, df$Blood_Pressure_Abnormality==1))

x <- df$salt_content_in_the_diet
h <- hist(x, breaks = 10, col = "light blue", xlab = "salt_content_in_the_diet", main = "Histogram with Normal Curve") 
xfit <- seq(min(x), max(x), length = 40) 
yfit <- dnorm(xfit, mean = mean(x), sd = sd(x)) 
yfit <- yfit * diff(h$mids[1:2]) * length(x) 
lines(xfit, yfit, col = "blue", lwd = 2)


# boxplot of alcohol_consumption_per_day by Blood_Pressure_Abnormality
boxplot(alcohol_consumption_per_day ~ Blood_Pressure_Abnormality,data = df, xlab = "Blood_Pressure_Abnormality", ylab = "alcohol_consumption_per_day")
t.test(subset(df$alcohol_consumption_per_day, df$Blood_Pressure_Abnormality==0), subset(df$alcohol_consumption_per_day, df$Blood_Pressure_Abnormality==1))

x <- df$alcohol_consumption_per_day
h <- hist(x, breaks = 10, col = "light blue", xlab = "alcohol_consumption_per_day", main = "Histogram with Normal Curve") 
xfit <- seq(min(x), max(x), length = 40) 
yfit <- dnorm(xfit, mean = mean(x), sd = sd(x)) 
yfit <- yfit * diff(h$mids[1:2]) * length(x) 
lines(xfit, yfit, col = "blue", lwd = 2)


# Gender
mytable <- xtabs( ~ Sex + Blood_Pressure_Abnormality, data=df)
CrossTable(mytable, prop.t = FALSE, prop.r = FALSE, prop.c = TRUE)
summary(mytable) 

# Pregnancy
mytable <- xtabs( ~ Pregnancy + Blood_Pressure_Abnormality, data=df)
CrossTable(mytable, prop.t = FALSE, prop.r = FALSE, prop.c = TRUE)
summary(mytable) 

# Smoking
mytable <- xtabs( ~ Smoking + Blood_Pressure_Abnormality, data=df)
CrossTable(mytable, prop.t = FALSE, prop.r = FALSE, prop.c = TRUE)
summary(mytable) 

# Level_of_Stress
mytable <- xtabs( ~ Level_of_Stress + Blood_Pressure_Abnormality, data=df)
CrossTable(mytable, prop.t = FALSE, prop.r = FALSE, prop.c = TRUE)
summary(mytable) 

# Chronic_kidney_disease
mytable <- xtabs( ~ Chronic_kidney_disease + Blood_Pressure_Abnormality, data=df)
CrossTable(mytable, prop.t = FALSE, prop.r = FALSE, prop.c = TRUE)
summary(mytable) 

# Adrenal_and_thyroid_disorders
mytable <- xtabs( ~ Adrenal_and_thyroid_disorders + Blood_Pressure_Abnormality, data=df)
CrossTable(mytable, prop.t = FALSE, prop.r = FALSE, prop.c = TRUE)
summary(mytable) 
