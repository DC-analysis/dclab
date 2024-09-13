require(stats);
require(lme4);

model_name <- "<MODEL_NAME>"
cat("OUTPUT model:", model_name, "#*#\n")

func_model <- "feature ~ group + (1 + group | repetition)"
func_nullmodel <- "feature ~ (1 + group | repetition)"

# These are the feature, group, and repetition arrays that are set by dclab
# via templates.
feature <- c(<FEATURES>)
group <- c(<GROUPS>)
repetition <- c(<REPETITIONS>)

data <- data.frame(feature, group, repetition)

if (model_name == "glmer+loglink") {
    Model <- glmer(func_model, data, family=Gamma(link='log'))
    NullModel <- glmer(func_nullmodel, data, family=Gamma(link='log'))
} else if (model_name == "lmer") {
    Model <- lmer(func_model, data)
    NullModel <- lmer(func_nullmodel, data)
} else {
    stop("Invalid model_name:", model_name)
}

# Anova analysis (increase verbosity by making models global)
# Using anova is a very conservative way of determining
# p values.
res_anova <- anova(Model, NullModel)
cat("OUTPUT r anova: ")
res_anova
cat("#*#\n")

pvalue <- res_anova$"Pr(>Chisq)"[2]
cat("OUTPUT anova p-value:", pvalue, "#*#\n")

model_summary <- summary(Model)
cat("OUTPUT r model summary:")
model_summary
cat("#*#\n")

model_coefficients <- coef(Model)
cat("OUTPUT r model coefficients:")
model_coefficients
cat("#*#\n")

fe_reps <- model_coefficients$repetition

effects <- data.frame(coef(model_summary))

fe_icept <- effects$Estimate[1]

fe_treat <- effects$Estimate[2]

if (model_name == "glmer+loglink") {
    # transform back from log
    fe_treat <- exp(fe_icept + fe_treat) - exp(fe_icept)
    fe_icept <- exp(fe_icept)
    fe_reps[, 2] = exp(fe_reps[, 1] + fe_reps[, 2]) - exp(fe_reps[, 1])
    fe_reps[, 1] = exp(fe_reps[, 1])
}

cat("OUTPUT fixed effects intercept:", fe_icept, "#*#\n")
cat("OUTPUT fixed effects treatment:", fe_treat, "#*#\n")
cat("OUTPUT fixed effects repetitions:")
fe_reps
cat("#*#\n")

# convergence

# convergence warnings in lme4
is_warning_generated <- function(m) {
  df <- summary(m)
  !is.null(df$optinfo$conv$lme4$messages) &&
           grepl('failed to converge', df$optinfo$conv$lme4$messages)
}
lme4_not_converged <- is_warning_generated(Model)

# convergence code by the optimizer
lme4l <- model_summary$optinfo$conv$lme4
if (length(lme4l) == 0) {
    # the optimizer probably does not know
    conv_code <- 0
} else if (is.null(lme4l$code)) {
    # NULL means 0
    conv_code <- 0
} else {
    conv_code <- lme4l$code
}

cat("OUTPUT model converged:", (conv_code == 0) && !lme4_not_converged, "#*#\n")
cat("OUTPUT lme4 messages:", lme4l$optinfo$conv$lme4$messages, "#*#\n")
