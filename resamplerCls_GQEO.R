resamplerCls_GQEO <- function(x, y, k = 5, ovRate = 1, check_planarity = TRUE) {
    library(FNN)

    x <- as.matrix(x)
    p <- ncol(x)
    class_names <- levels(y)
    n_classes <- sapply(class_names, function(cl) sum(y == cl))
    k_class <- length(class_names)

    n_needed <- round((max(n_classes) - n_classes) * ovRate)

    x_syn_list <- lapply(1:k_class, function(i) matrix(NA, nrow = 0, ncol = p))
    y_syn_list <- lapply(1:k_class, function(i) factor(c(), levels = levels(y)))

    for (i_class in 1:k_class) {
        if (n_needed[i_class] == 0)
            next

        x_min <- x[y == class_names[i_class], , drop = FALSE]
        n_min <- nrow(x_min)
        dists <- as.matrix(dist(x_min))
        knn_graph <- get.knn(x_min, k = k)$nn.index

        quadrilaterals <- list()
        for (i in 1:n_min) {
            neighbors <- knn_graph[i, ]
            combn_neighbors <- combn(neighbors, 3, simplify = FALSE)
            for (triplet in combn_neighbors) {
                quad <- sort(unique(c(i, triplet)))
                if (length(quad) == 4) {
                  quadrilaterals[[length(quadrilaterals) + 1]] <- quad
                }
            }
        }

        # Deduplicate
        quadrilaterals <- unique(lapply(quadrilaterals, function(q) paste(sort(q), collapse = "-")))
        quadrilaterals <- lapply(quadrilaterals, function(q) as.integer(strsplit(q, "-")[[1]]))

        if (check_planarity) {
            is_planar <- function(q, distmat) {
                edges <- c(distmat[q[1], q[2]], distmat[q[2], q[3]], distmat[q[3], q[4]], distmat[q[4],
                  q[1]])
                for (i in 1:4) {
                  if (sum(edges[-i]) <= edges[i])
                    return(FALSE)
                }
                return(TRUE)
            }
            quadrilaterals <- Filter(function(q) is_planar(q, dists), quadrilaterals)
        }

        n_gen <- n_needed[i_class]
        selected_quads <- sample(quadrilaterals, size = n_gen, replace = TRUE)
        synth <- matrix(NA, nrow = n_gen, ncol = p)

        for (i in seq_len(n_gen)) {
            q <- selected_quads[[i]]
            X <- x_min[q, , drop = FALSE]

            eta <- runif(1)
            xi <- 1 - eta

            a1 <- 0.25 * (1 - xi) * (1 - eta)
            a2 <- 0.25 * (1 + xi) * (1 - eta)
            a3 <- 0.25 * (1 + xi) * (1 + eta)
            a4 <- 0.25 * (1 - xi) * (1 + eta)

            synth[i, ] <- a1 * X[1, ] + a2 * X[2, ] + a3 * X[3, ] + a4 * X[4, ]
        }

        x_syn_list[[i_class]] <- synth
        y_syn_list[[i_class]] <- factor(rep(class_names[i_class], n_gen), levels = levels(y))
    }

    x_syn <- do.call(rbind, x_syn_list)
    y_syn <- do.call(c, y_syn_list)
    x_new <- rbind(x, x_syn)
    y_new <- c(y, y_syn)

    return(list(x_new = x_new, y_new = y_new, x_syn = x_syn, y_syn = y_syn))
}

# dat <- datasets_classification_binary$banana x <- dat[,1:2] y <- dat[,3] m_deepSMOTE <-
# resamplerCls_GQEO(x = x, y = y) plot(x, col = y) points(x = m_deepSMOTE$x_syn, col = 'green')
