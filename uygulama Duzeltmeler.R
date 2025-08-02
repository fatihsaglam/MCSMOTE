library(writexl)
library(doParallel)
library(foreach)
library(pROC)
library(MLmetrics)
library(PRROC)
library(FNN)
library(data.table)
library(SMOTEWB)
library(randomForest)
library(e1071)
library(kernlab)
library(nnet)
library(rpart)
library(kknn)
library(naivebayes)
library(caret)
library(yardstick)
library(reticulate)


nn <- import(module = "torch.nn")
torch <- import(module = "torch")

# saveRDS(object = datasets_classification_binary, file = "datasets_classification_binary.rds")
datasets_classification_binary <- readRDS(file = "datasets_classification_binary.rds")

source(file = "resamplerCls_AWGAN.R")
source(file = "resamplerCls_deepSMOTE.R")
source(file = "resamplerCls_GQEO.R")
source(file = "TMMCSMOTE.R")

resamplers_all <- list(
  NORES = list(
    resampler = function(x, y) {
      return(list(
        x_new = x,
        y_new = y,
        x_syn = NULL,
        y_syn = NULL
      ))
    },
    hyperParList = expand.grid()
  ),
  ROS = list(
    resampler = SMOTEWB::ROS,
    hyperParList = expand.grid(
      ovRate = c(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2)
    )
  ),
  RWO = list(
    resampler = SMOTEWB::RWO,
    hyperParList = expand.grid(
      ovRate = c(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2)
    )
  ),
  SMOTE = list(
    resampler = SMOTEWB::SMOTE,
    hyperParList = expand.grid(
      k = c(3, 5, 7),
      ovRate = c(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2)
    )
  ),
  ADASYN = list(
    resampler = SMOTEWB::ADASYN,
    hyperParList = expand.grid(
      k = c(3, 5, 7),
      ovRate = c(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2)
    )
  ),
  BLSMOTE = list(
    resampler = SMOTEWB::BLSMOTE,
    hyperParList = expand.grid(
      k1 = c(3, 5, 7),
      k2 = c(3, 5, 7),
      ovRate = c(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2)
    )
  ),
  SMOTEWB = list(
    resampler = SMOTEWB::SMOTEWB,
    hyperParList = expand.grid(
      n_weak = c(25, 50, 100),
      ovRate = c(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2)
    )
  ),
  GSMOTE = list(
    resampler = SMOTEWB::GSMOTE,
    hyperParList = expand.grid(
      k = c(3, 5, 7),
      alpha_trunc = c(-1, -0.5, 0, 0.5, 1),
      alpha_def = c(0, 0.5, 1),
      ovRate = c(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2)
    )
  ),
  deepSMOTE = list(
    resampler = resamplerCls_deepSMOTE,
    hyperParList = expand.grid(
      k = c(3, 5, 7),
      epochs = c(50L, 100L, 200L),
      hidden_dim = c(16L, 32L),
      latent_dim = c(16L, 32L),
      ovRate = c(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2)
    )
  ),
  GQED = list(
    resampler = resamplerCls_GQEO,
    hyperParList = expand.grid(
      k = c(3, 5, 7),
      ovRate = c(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2)
    )
  ),
  MCSMOTE = list(
    resampler = resamplerCls_MCSMOTE,
    hyperParList = expand.grid(
      sigma = c(0.1, 0.25, 0.5, 0.75, 1),
      thresh_qntl = c(0.1, 0.25, 0.5, 0.75, 1),
      ovRate = c(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2)
    )
  )
)

results_file <- "Results_Resampling_Experiments.xlsx"
if (!file.exists(results_file)) {
  write_xlsx(x = list(
    Results = data.frame(
      Dataset = character(),
      Classifier = character(),
      Resampler = character(),
      Fold = integer(),
      BestHyperparResampler = character(),
      BalancedAccuracy = numeric(),
      MCC = numeric(),
      GMean = numeric(),
      F1 = numeric(),
      ROC_AUC = numeric(),
      PR_AUC = numeric(),
      Runtime = numeric()
    )
  ), path = results_file)
}

Gmean <- function(pred, true) {
  tab <- table(factor(pred, levels = levels(factor(true))),
               factor(true, levels = levels(factor(true))))
  if (nrow(tab) != 2 || ncol(tab) != 2) return(NA)
  sensitivity <- tab[2, 2] / sum(tab[, 2])
  specificity <- tab[1, 1] / sum(tab[, 1])
  sqrt(sensitivity * specificity)
}

MCC <- function(pred, true) {
  tab <- table(factor(pred, levels = levels(factor(true))),
               factor(true, levels = levels(factor(true))))
  if (nrow(tab) != 2 || ncol(tab) != 2) return(NA)
  TP <- tab[2, 2]
  TN <- tab[1, 1]
  FP <- tab[2, 1]
  FN <- tab[1, 2]
  numerator <- (TP * TN) - (FP * FN)
  # denominator <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  denominator <- sqrt(TP + FP) * sqrt(TP + FN) * sqrt(TN + FP) * sqrt(TN + FN)
  if (denominator == 0) return(0)
  return(numerator / denominator)
}

evaluate_metrics <- function(y_true, y_pred, y_prob) {
  y_true <- factor(y_true)
  y_pred <- factor(y_pred, levels = levels(y_true))
  class_counts <- table(y_true)
  positive_class <- names(class_counts)[which.min(class_counts)]
  y_true <- factor(y_true, levels = c(setdiff(levels(y_true), positive_class), positive_class))
  y_pred <- factor(y_pred, levels = levels(y_true))
  ba <- (mean((y_pred == y_true)[y_true == positive_class]) + mean((y_pred == y_true)[y_true != positive_class]))/2
  mcc <- MCC(pred = y_pred, true = y_true)
  gm <- Gmean(y_pred, y_true)
  f1 <- MLmetrics::F1_Score(y_true = y_true, y_pred = y_pred, positive = positive_class)
  if (is.nan(f1)) {
    f1 <- 0
  }
  roc <- tryCatch({
    as.numeric(pROC::auc(pROC::roc(response = y_true, predictor = y_prob,
                                   levels = levels(y_true), direction = "<", quiet = TRUE)))
  }, error = function(e) NA)
  pr <- tryCatch({
    pr.curve(scores.class0 = y_prob[y_true == positive_class],
             scores.class1 = y_prob[y_true != positive_class])$auc.integral
  }, error = function(e) NA)
  c(BalancedAccuracy = ba, MCC = mcc, GMean = gm, F1 = f1, ROC_AUC = roc, PR_AUC = pr)
}

preprocess_data <- function(x_train, x_test) {
  vars <- apply(x_train, 2, var)
  keep <- vars >= 1e-4
  x_train <- x_train[, keep, drop = FALSE]
  x_test <- x_test[, keep, drop = FALSE]
  means <- apply(x_train, 2, mean)
  sds <- apply(x_train, 2, sd)
  sds[sds == 0] <- 1
  x_train_scaled <- scale(x_train, center = means, scale = sds)
  x_test_scaled  <- scale(x_test, center = means, scale = sds)
  list(x_train = x_train_scaled, x_test = x_test_scaled)
}

datasets <- datasets_classification_binary
cores_to_use <- min(10, parallel::detectCores() - 1)
cl <- makeCluster(cores_to_use)

all_datasets_results <- list()

# clusterEvalQ(cl = cl, expr = {
#   library(SMOTEWB)
#   library(randomForest)
#   library(e1071)
#   library(kernlab)
#   library(nnet)
#   library(rpart)
#   library(naivebayes)
#   library(caret)
#   library(yardstick)
#   library(reticulate)
#   library(PRROC)
#   nn <- import(module = "torch.nn")
#   torch <- import(module = "torch")
# })

for (i_dataset in c(140:141)) {
  dataset_name <- names(datasets)[i_dataset]
  cat("\nProcessing dataset:", dataset_name, "\n")
  
  data <- datasets[[dataset_name]]
  x <- data[, -ncol(data)]
  y <- data[, ncol(data)]
  folds <- createFolds(y, k = 10, list = TRUE)
  classifier_list <- c("rpart", "rf", "svm", "nb", "nn", "glm", "knn")

  # clusterExport(
  #   cl = cl,
  #   varlist = c(
  #     "resamplers_all",
  #     "classifier_list",
  #     "folds",
  #     "x",
  #     "y",
  #     "preprocess_data",
  #     "evaluate_metrics",
  #     "Gmean",
  #     "MCC",
  #     "dataset_name",
  #     "gaussian_kernel",
  #     "sinkhorn_normalize"))
  
  # clusterExport(cl, varlist = c("resamplers_all"), envir = environment())
  
  results_rows <- lapply(names(resamplers_all),  function(resampler_name) {
  # results_rows <- parLapply(cl = cl, X = names(resamplers_all), fun = function(resampler_name) {
    cat("\n-Processing resampler:", resampler_name, "\n")
    set.seed(42)
    # resampler_name <- names(resamplers_all)[11]
    resampler_info <- resamplers_all[[resampler_name]]
    grid <- resampler_info$hyperParList
    sampled_params <- if (ncol(grid) == 0 || nrow(grid) == 0) {
      list(list())  # one empty list: no hyperparameters
    } else {
      rows <- if (nrow(grid) > 10) sample(nrow(grid), 10) else seq_len(nrow(grid))
      lapply(rows, function(i) as.list(grid[i,,drop = FALSE]))
    }
    local_rows <- list()
    
    for (clf_name in classifier_list) {
      cat("\n--classifier resampler:", clf_name, "\n")
      best_auc <- -Inf
      best_param <- NULL
      
      for (param in sampled_params) {
        aucs <- c()
        for (f in 1:3) {
          idx_test <- folds[[f]]
          idx_train <- setdiff(seq_len(nrow(x)), idx_test)
          prep <- preprocess_data(x[idx_train, ], x[idx_test, ])
          
          res <- do.call(resampler_info$resampler, c(list(x = prep$x_train, y = y[idx_train]), param))
          try({
            model <- switch(clf_name,
                            rf    = randomForest(x = res$x_new, y = as.factor(res$y_new), ntree = 100),
                            svm   = ksvm(x = as.matrix(res$x_new), y = as.factor(res$y_new), kernel = "rbfdot", prob.model = TRUE),
                            knn   = caret::knn3(x = res$x_new, y = as.factor(res$y_new), k = 5),
                            nn    = nnet(x = res$x_new, y = class.ind(as.factor(res$y_new)), size = 5, decay = 0.01, maxit = 200, trace = FALSE),
                            rpart = rpart(class ~ ., data = data.frame(class = as.factor(res$y_new), res$x_new)),
                            nb    = naive_bayes(x = res$x_new, y = as.factor(res$y_new)),
                            glm   = glm(as.factor(res$y_new) ~ ., data = data.frame(res$x_new), family = binomial())
            )
            
            class_counts <- table(y)
            positive_class <- names(class_counts)[which.min(class_counts)]
            probs <- switch(clf_name,
                            rf    = predict(model, prep$x_test, type = "prob"),
                            svm   = predict(model, as.matrix(prep$x_test), type = "probabilities"),
                            knn   = predict(model, prep$x_test, type = "prob"),
                            nn    = predict(model, prep$x_test),
                            rpart = predict(model, as.data.frame(prep$x_test)),
                            nb    = predict(model, prep$x_test, type = "prob"),
                            glm   = {
                              prob = data.frame(`1` = predict(model, newdata = as.data.frame(prep$x_test), type = "response"))
                              colnames(prob) <- positive_class
                              prob
                            }
            )
            positive_col <- which(colnames(probs) == positive_class)
            y_prob <- as.vector(probs[, positive_col])
            auc_val <- as.numeric(pROC::auc(pROC::roc(y[idx_test], y_prob, quiet = TRUE)))
            aucs <- c(aucs, auc_val)
          })
          
        }
        mean_auc <- mean(aucs, na.rm = TRUE)
        if (!is.na(mean_auc) && mean_auc > best_auc) {
          best_auc <- mean_auc
          best_param <- param
        }
      }
      
      # 10-fold CV
      for (f in 1:10) {
        idx_test <- folds[[f]]
        idx_train <- setdiff(seq_len(nrow(x)), idx_test)
        prep <- preprocess_data(x[idx_train, ], x[idx_test, ])
        t1 <- Sys.time()
        res <- do.call(resampler_info$resampler, c(list(x = prep$x_train, y = y[idx_train]), best_param))
        t2 <- Sys.time()
        runtime <- as.numeric(difftime(t2, t1, units = "secs"))
        model <- switch(clf_name,
                        rf = randomForest(x = res$x_new, y = as.factor(res$y_new), ntree = 100),
                        svm = ksvm(x = as.matrix(res$x_new), y = as.factor(res$y_new), kernel = "rbfdot", prob.model = TRUE),
                        knn = caret::knn3(x = res$x_new, y = as.factor(res$y_new), k = 5),
                        nn  = nnet(x = res$x_new, y = class.ind(as.factor(res$y_new)), size = 5, decay = 0.01, maxit = 200, trace = FALSE),
                        rpart = rpart(class ~ ., data = data.frame(class = as.factor(res$y_new), res$x_new)),
                        nb = naive_bayes(x = res$x_new, y = as.factor(res$y_new)),
                        glm = glm(as.factor(res$y_new) ~ ., data = data.frame(res$x_new), family = binomial())
        )
        class_counts <- table(y)
        positive_class <- names(class_counts)[which.min(class_counts)]
        probs <- switch(clf_name,
                        rf    = predict(model, prep$x_test, type = "prob"),
                        svm   = predict(model, as.matrix(prep$x_test), type = "probabilities"),
                        knn   = predict(model, prep$x_test, type = "prob"),
                        nn    = predict(model, prep$x_test),
                        rpart = predict(model, as.data.frame(prep$x_test)),
                        nb    = predict(model, prep$x_test, type = "prob"),
                        glm   = {
                          prob = data.frame(`1` = predict(model, newdata = as.data.frame(prep$x_test), type = "response"))
                          colnames(prob) <- positive_class
                          prob
                        }        )
        positive_col <- which(colnames(probs) == positive_class)
        y_prob <- as.vector(probs[, positive_col])
        y_pred <- factor(ifelse(y_prob > 0.5, levels(as.factor(y))[2], levels(as.factor(y))[1]), levels = levels(as.factor(y)))
        metrics <- evaluate_metrics(y_true = y[idx_test], y_pred = y_pred, y_prob = y_prob)

        best_param_str <- if (length(best_param) == 0) {
          ""
        } else {
          paste(names(best_param), unlist(best_param), sep = "=", collapse = ";")
        }

        row <- data.frame(Dataset = dataset_name,
                          Classifier = clf_name,
                          Resampler = resampler_name,
                          Fold = f,
                          BestHyperparResampler = best_param_str,
                          t(metrics),
                          Runtime = runtime)

        local_rows[[length(local_rows) + 1]] <- row
      }
    }
    rslt <- do.call(rbind, local_rows)
    rslt
  })
  results_rows <- do.call(rbind, results_rows)
  for (j in 1:ncol(results_rows)) {
    results_rows[,j] <- unlist(results_rows[,j])
  }
  all_datasets_results[[dataset_name]] <- results_rows  # ✅ Step 2
  
  final_df <- do.call(rbind, all_datasets_results)
  results_file <- "Results_Resampling_Experiments.xlsx"
  for (j in 1:ncol(final_df)) {
    final_df[,j] <- unlist(final_df[,j])
  }
  
  if (file.exists(results_file)) {
    
    previous_results <- readxl::read_xlsx(results_file)
    combined_results <- rbind(previous_results, results_rows)
  } else {
    combined_results <- results_rows
  }
  writexl::write_xlsx(combined_results, path = results_file)
}


stopCluster(cl)

