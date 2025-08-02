# library(FatihResearch)
# library(reticulate)
# np <- reticulate::import("numpy")
# torch <- reticulate::import("torch")
# nn <- import("torch.nn", convert = TRUE)

resamplerCls_deepSMOTE <- function(x, y, k = 5L, lr = 0.01, batch_size = 32L, epochs = 100L, hidden_dim = 16L,
    latent_dim = 4L, ovRate = 1) {
    Encoder <- PyClass("Encoder", inherit = nn$Module, defs = list(`__init__` = function(self, input_dim,
        hidden_dim, latent_dim) {
        super()$`__init__`()
        self$fc1 <- nn$Linear(input_dim, hidden_dim)
        self$relu <- nn$ReLU()
        self$fc2 <- nn$Linear(hidden_dim, latent_dim)
        NULL
    }, forward = function(self, x) {
        x <- self$fc1(x)
        x <- self$relu(x)
        x <- self$fc2(x)
        return(x)
    }))

    Decoder <- PyClass("Decoder", inherit = nn$Module, defs = list(`__init__` = function(self, latent_dim,
        hidden_dim, output_dim) {
        super()$`__init__`()
        self$fc1 <- nn$Linear(latent_dim, hidden_dim)
        self$relu <- nn$ReLU()
        self$fc2 <- nn$Linear(hidden_dim, output_dim)
        NULL
    }, forward = function(self, x) {
        x <- self$fc1(x)
        x <- self$relu(x)
        x <- self$fc2(x)
        return(x)
    }))

    permute_order <- function(x) {
        idx <- torch$randperm(x$size(1L))
        return(x[, idx])
    }

    reconstruction_loss <- function(pred, target) {
        return(torch$mean((pred - target)$pow(2L)))
    }

    train_deepsmote <- function(data_loader, encoder, decoder, optimizer, epochs) {
        for (epoch in 1:epochs) {
            for (batch in data_loader) {
                optimizer$zero_grad()
                EB <- encoder(batch)
                DB <- decoder(EB)

                CD <- batch
                ES <- encoder(CD)
                PE <- permute_order(ES)
                DP <- decoder(PE)

                PL <- reconstruction_loss(DP, CD)
                RL <- reconstruction_loss(DB, batch)

                TL <- RL + PL
                TL$backward()
                optimizer$step()
            }
        }
    }

    generate_samples <- function(x_pos, encoder, decoder, n_samples, k = 5L) {
        embeddings <- encoder(x_pos)
        x_main <- embeddings$detach()$numpy()

        NN_main2main <- FNN::get.knnx(data = x_main, query = x_main, k = k + 1)$nn.index[, -1]

        x_syn <- matrix(data = NA, nrow = 0, ncol = ncol(x_main))
        n_main <- nrow(x_main)

        while (nrow(x_syn) < n_samples) {

            i_sample <- sample(1:n_main, size = 1)
            x_main_selected <- x_main[i_sample, , drop = FALSE]
            x_target <- x_main[sample(NN_main2main[i_sample, ], size = 1), , drop = FALSE]
            r <- runif(1)

            x_syn <- rbind(x_syn, x_main_selected + r * (x_target - x_main_selected))
        }

        x_syn_tensor <- torch$tensor(as.matrix(x_syn), dtype = torch$float32)

        synthetic_samples <- decoder(x_syn_tensor)
        return(synthetic_samples)
    }

    create_batches <- function(tensor_data, batch_size) {
        n <- tensor_data$size(0L)
        idx <- torch$randperm(n)
        batches <- list()
        for (i in seq(1, n, by = batch_size)) {
            end <- min(i + batch_size - 1, n)
            batches[[length(batches) + 1]] <- tensor_data[idx[i:end], ]
        }
        return(batches)
    }

    p <- ncol(x)
    x <- as.matrix(x)
    class_names <- levels(y)
    n_classes <- sapply(class_names, function(m) sum(y == m))
    k_class <- length(class_names)

    n_needed <- round((max(n_classes) - n_classes) * ovRate)


    x_syn_list <- lapply(1:k_class, function(m) matrix(NA, nrow = 0, ncol = p))
    y_syn_list <- lapply(1:k_class, function(m) factor(c(), levels = levels(y)))

    for (i_class in 1:k_class) {
        if (n_needed[i_class] == 0) {
            next
        }

        x_main <- x[y == class_names[i_class], , drop = FALSE]
        x_main_tensor <- torch$tensor(as.matrix(x_main), dtype = torch$float32)

        batches <- create_batches(tensor_data = x_main_tensor, batch_size)

        encoder <- Encoder(p, hidden_dim, latent_dim)
        decoder <- Decoder(latent_dim, hidden_dim, p)

        params <- c(iterate(encoder$parameters()), iterate(decoder$parameters()))
        optim <- torch$optim$Adam(params, lr = lr)

        train_deepsmote(batches, encoder, decoder, optim, epochs)

        x_syn_temp <- generate_samples(x_main_tensor, encoder, decoder, n_samples = n_needed[i_class], k = k)
        x_syn_list[[i_class]] <- x_syn_temp$detach()$numpy()
        y_syn_temp <- factor(rep(class_names[i_class], nrow(x_syn_list[[i_class]])), levels = levels(y))
        y_syn_list[[i_class]] <- y_syn_temp
    }

    x_syn <- do.call(rbind, x_syn_list)
    y_syn <- do.call(c, y_syn_list)

    x_new <- rbind(as.matrix(x), x_syn)
    y_new <- c(y, y_syn)

    return(list(x_new = x_new, y_new = y_new, x_syn = x_syn, y_syn = y_syn))
}

# dat <- datasets_classification_binary$banana x <- dat[,1:2] y <- dat[,3] m_deepSMOTE <-
# resamplerCls_deepSMOTE(x = x, y = y, batch_size = 64L, epochs = 100L) plot(x, col = y) points(x =
# m_deepSMOTE$x_syn, col = 'green')
