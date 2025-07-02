
# --------------------------------------------------
# Tutorial/Lab 1: Linear Regression on the Diabetes Dataset
# --------------------------------------------------

# 1) Packages and Setup
# install.packages(c("lars","caret","ggplot2","corrplot"))
library(lars)      # for diabetes data
library(caret)     # for train/test split
library(ggplot2)   # for plotting
library(corrplot)  # for correlation heatmap

set.seed(1234)

# 2) Load & Prepare Data
data("diabetes", package="lars")        # loads diabetes$x and diabetes$y
df <- as.data.frame(diabetes$x)
df$target <- diabetes$y

# Confirm dimensions
cat("Data frame dimensions:", dim(df), "\n")
head(df)

# 3) Exploratory Data Analysis (EDA)
corr_mat <- cor(df)
corrplot(corr_mat,
         method    = "color",
         type      = "upper",
         tl.cex    = 0.8,
         addCoef.col = "black",
         diag      = FALSE,
         title     = "Correlation Matrix (Diabetes)")

# 4) Train–Test Split (70/30)
train_idx <- createDataPartition(df$target, p = 0.8, list = FALSE)
df_train  <- df[train_idx, ]
df_test   <- df[-train_idx, ]

X_train    <- as.matrix(df_train[, setdiff(names(df_train), "target")])
y_train    <- df_train$target
X_test     <- as.matrix(df_test[, setdiff(names(df_test), "target")])
y_test     <- df_test$target

# Add intercept column
X_train_b  <- cbind(Intercept = 1, X_train)
X_test_b   <- cbind(Intercept = 1, X_test)


# 6) Closed-form OLS solution
beta_closed <- solve(t(X_train_b) %*% X_train_b) %*% (t(X_train_b) %*% y_train)
beta_closed <- as.vector(beta_closed)

# 7) lm() solution
lm_fit      <- lm(target ~ ., data = df_train)
beta_lm    <- coef(lm_fit)

# 8) Compare Coefficients
# Assign names so that rbind preserves column names
col_names   <- colnames(X_train_b)
names(beta_closed) <- col_names
names(beta_lm)      <- col_names

coef_table <- rbind(
  ClosedForm       = beta_closed,
  LM               = beta_lm
)
cat("\nCoefficient estimates (rows: methods; cols: Intercept + features):\n")
print(round(coef_table, 4))

# 10) Compute R^2 on Train & Test Sets
r2_score <- function(y, yhat) {
  1 - sum((y - yhat)^2) / sum((y - mean(y))^2)
}

# Predictions
y_train_pred_cf <- X_train_b %*% beta_closed
y_test_pred_cf  <- X_test_b  %*% beta_closed

y_train_pred_lm <- predict(lm_fit, newdata = df_train)
y_test_pred_lm  <- predict(lm_fit, newdata = df_test)


# R^2 values
r2_train_cf <- r2_score(y_train, y_train_pred_cf)
r2_test_cf  <- r2_score(y_test,  y_test_pred_cf)
r2_train_lm <- r2_score(y_train, y_train_pred_lm)
r2_test_lm  <- r2_score(y_test,  y_test_pred_lm)


r2_results <- data.frame(
  Model     = c("ClosedForm","LM"),
  R2_Train  = c(r2_train_cf,  r2_train_lm),
  R2_Test   = c(r2_test_cf,   r2_test_lm)
)

cat("\nR^2 scores on training and testing data:\n")
print(r2_results)

# 11) Overfitting & Next Steps
cat("\nNote on Overfitting:\n")
cat(" If R^2_Test is much lower than R^2_Train, the model may be overfitting.\n")
cat(" Regularization (e.g., Ridge, Lasso) can mitigate overfitting; we cover this next week.\n")
