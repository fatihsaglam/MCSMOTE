resamplerCls_AWGAN <- function(x, y, lr = 1e-04, batch_size = 64L, epochs = 100L, hidden_dim = 32L, noise_dim = 16L,
    lambda_gp = 10, similarity_threshold = 0.8, ovRate = 1) {
    library(reticulate)
    torch <- import("torch")
    nn <- import("torch.nn", convert = TRUE)
    F <- import("torch.nn.functional")

    Generator <- PyClass("Generator", inherit = nn$Module, defs = list(`__init__` = function(self, noise_dim,
        hidden_dim, output_dim) {
        super()$`__init__`()
        self$fc1 <- nn$Linear(noise_dim, hidden_dim)
        self$relu <- nn$ReLU()
        self$fc2 <- nn$Linear(hidden_dim, output_dim)
        NULL
    }, forward = function(self, z) {
        self$fc2(self$relu(self$fc1(z)))
    }))

    Discriminator <- PyClass("Discriminator", inherit = nn$Module, defs = list(`__init__` = function(self,
        input_dim, hidden_dim) {
        super()$`__init__`()
        self$fc1 <- nn$Linear(input_dim, hidden_dim)
        self$relu <- nn$ReLU()
        self$fc2 <- nn$Linear(hidden_dim, 1L)
        NULL
    }, forward = function(self, x) {
        self$fc2(self$relu(self$fc1(x)))
    }))

    compute_gp <- function(D, real, fake) {
        alpha <- torch$rand(real$size(0L), 1L)$expand_as(real)
        interpolates <- alpha * real + (1 - alpha) * fake
        interpolates$requires_grad_(TRUE)
        d_interpolates <- D(interpolates)
        grad_outputs <- torch$ones_like(d_interpolates)
        gradients <- torch$autograd$grad(outputs = d_interpolates, inputs = interpolates, grad_outputs = grad_outputs,
            create_graph = TRUE, retain_graph = TRUE, only_inputs = TRUE)[[1]]
        gradients <- gradients$view(gradients$size(0L), -1L)
        gp <- ((gradients$norm(2L, dim = 1L) - 1L)^2)$mean()
        return(gp)
    }

    train_awgan <- function(x_real, G, D, optim_G, optim_D, epochs, batch_size) {
        n <- x_real$size(0L)
        for (epoch in 1:epochs) {
            idx <- torch$randperm(n)
            for (i in seq(1, n, by = batch_size)) {
                end <- min(i + batch_size - 1, n)
                real_batch <- x_real[idx[i:end], ]
                b_size <- real_batch$size(0L)
                for (t in 1:5) {
                  z <- torch$randn(b_size, noise_dim)
                  fake_batch <- G(z)$detach()
                  optim_D$zero_grad()
                  loss_D <- -(D(real_batch)$mean() - D(fake_batch)$mean()) + lambda_gp * compute_gp(D, real_batch,
                    fake_batch)
                  loss_D$backward()
                  optim_D$step()
                }
                z <- torch$randn(b_size, noise_dim)
                fake_batch <- G(z)
                optim_G$zero_grad()
                loss_G <- -D(fake_batch)$mean()
                loss_G$backward()
                optim_G$step()
            }
        }
    }

    cosine_filter <- function(x_real, x_gen, threshold) {
        sim_matrix <- F$cosine_similarity(x_gen$unsqueeze(1L), x_real$unsqueeze(0L), dim = 2L)
        max_sim <- sim_matrix$max(dim = 1L)[[0]]
        keep <- (max_sim >= threshold)
        return(x_gen[keep, ])
    }

    x <- as.matrix(x)
    p <- ncol(x)
    class_names <- levels(y)
    k_class <- length(class_names)
    n_classes <- sapply(class_names, function(cl) sum(y == cl))
    n_needed <- as.integer(round((max(n_classes) - n_classes) * ovRate))

    x_syn_list <- list()
    y_syn_list <- lapply(1:k_class, function(m) {
        factor(c(), levels = levels(y))
    })

    for (i in seq_along(class_names)) {
        if (n_needed[i] == 0)
            next
        x_min <- x[y == class_names[i], , drop = FALSE]
        x_tensor <- torch$tensor(x_min, dtype = torch$float32)

        G <- Generator(noise_dim, hidden_dim, p)
        D <- Discriminator(p, hidden_dim)
        optim_G <- torch$optim$Adam(G$parameters(), lr = lr, betas = tuple(0.5, 0.9))
        optim_D <- torch$optim$Adam(D$parameters(), lr = lr, betas = tuple(0.5, 0.9))

        train_awgan(x_tensor, G, D, optim_G, optim_D, epochs, batch_size)

        n_gen <- n_needed[i] * 2L
        x_gen <- G(torch$randn(n_gen, noise_dim))
        x_gen_filtered <- cosine_filter(x_tensor, x_gen, similarity_threshold)

        if (x_gen_filtered$size(0L) >= n_needed[i]) {
            x_final <- x_gen_filtered[1:n_needed[i], ]
        } else {
            pad <- n_needed[i] - x_gen_filtered$size(0L)
            extra <- G(torch$randn(pad, noise_dim))
            x_final <- torch$cat(list(x_gen_filtered, extra), dim = 0L)
        }

        x_syn_list[[i]] <- x_final$detach()$numpy()
        y_syn_list[[i]] <- factor(rep(class_names[i], n_needed[i]), levels = levels(y))
    }


    x_syn <- do.call(rbind, x_syn_list)
    y_syn <- do.call(c, y_syn_list)
    x_new <- rbind(x, x_syn)
    y_new <- c(y, y_syn)

    return(list(x_new = x_new, y_new = y_new, x_syn = x_syn, y_syn = y_syn))
}
# dat <- datasets_classification_binary$banana x <- dat[,1:2] y <- dat[,3] m_deepSMOTE <-
# resamplerCls_AWGAN(x = x, y = y) plot(x, col = y) points(x = m_deepSMOTE$x_syn, col = 'green')
