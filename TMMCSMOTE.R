sinkhorn_normalize <- function(mat, tol = 1e-04, max_iter = 100) {
    for (i in 1:max_iter) {
        mat <- t(apply(mat, 1, function(m) m/sum(m)))
        mat <- t(apply(mat, 2, function(m) m/sum(m)))

        if (all(abs(rowSums(mat) - 1) < tol) && all(abs(colSums(mat) - 1) < tol)) {
            break
        }
    }
    return(mat)
}
gaussian_kernel <- function(dist_mat, sigma = NULL, thresh_qntl = 0.1) {
    qntl <- apply(dist_mat, 2, function(m) quantile(m, thresh_qntl))
    sapply(1:ncol(dist_mat), function(m) {
        exp(-dist_mat[, m]^2/(2 * qntl[m]^2))
    })
    # exp(-dist_mat^2 / (2 * sigma^2))
}

resamplerCls_MCSMOTE <- function(x, y, sigma = 0.1, thresh_qntl = 0.1, feature_weights = NULL, noiseBlocking = FALSE,
    ovRate = 1) {
    p <- ncol(x)
    n <- nrow(x)

    if (is.null(feature_weights)) {
        feature_weights <- rep(1, p)
    } else {
        if (length(feature_weights) < p) {
            stop("length of feature weights must be equal to the number of features")
        }
        if (!is.numeric(feature_weights)) {
            stop("feature_weights must be numeric")
        }

        feature_weights <- feature_weights/sum(feature_weights) * p
    }

    var_names <- colnames(x)
    x <- as.matrix(x)

    class_names <- levels(y)
    class_pos <- names(which.min(table(y)))
    class_neg <- class_names[class_names != class_pos]

    x_pos <- x[y == class_pos, , drop = FALSE]
    x_neg <- x[y == class_neg, , drop = FALSE]

    n_pos <- nrow(x_pos)
    n_neg <- nrow(x_neg)

    M_weight_pos <- t(replicate(n_pos, c(feature_weights)))

    dist_pos2pos <- Rfast::Dist(M_weight_pos * x_pos)
    weights_pos2pos <- gaussian_kernel(dist_pos2pos, sigma = sigma, thresh_qntl)
    weights_pos2pos_normalized <- sinkhorn_normalize(weights_pos2pos)

    if (noiseBlocking) {
        M_weight_neg <- t(replicate(n_neg, c(feature_weights)))
        dist_pos2neg <- Rfast::dista(xnew = M_weight_pos * x_pos, x = M_weight_neg * x_neg)
        weights_pos2neg <- gaussian_kernel(dist_pos2neg, sigma = sigma, thresh_qntl)
        weights_pos2pos_normalized <- sinkhorn_normalize(weights_pos2neg)

        i_safe <- !(rowMeans(dist_pos2neg) < rowMeans(dist_pos2pos))
    } else {
        i_safe <- rep(TRUE, n_pos)
    }

    i_nn_pos2pos <- RANN::nn2(data = M_weight_pos * x_pos, query = M_weight_pos * x_pos, k = 2)$nn.idx[,
        -1, drop = FALSE]

    n_needed <- round((n_neg - n_pos) * ovRate)

    x_syn <- matrix(data = NA, nrow = 0, ncol = p)

    repeat {
        if (nrow(x_syn) >= n_needed) {
            break
        }
        i_selected <- sample((1:n_pos)[i_safe], size = 1)

        i_center <- i_nn_pos2pos[i_selected, ]
        i_target <- sample((1:n_pos)[i_safe], size = 1, prob = (weights_pos2pos_normalized[i_center, ])[i_safe])
        r <- runif(1)
        x_syn_step <- x_pos[i_center, ] - r * (x_pos[i_center, ] - x_pos[i_target, ])
        x_syn <- rbind(x_syn, x_syn_step)
    }

    y_syn <- factor(rep(class_pos, n_needed), levels = class_names, labels = class_names)

    x_new <- rbind(x, x_syn)
    y_new <- c(y, y_syn)
    rownames(x_new) <- NULL
    colnames(x_new) <- var_names
    return(list(x_new = x_new, y_new = y_new, x_syn = x_syn, y_syn = y_syn))
}
