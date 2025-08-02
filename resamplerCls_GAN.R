library(FatihResearch)
library(reticulate)
np <- reticulate::import("numpy")
torch <- reticulate::import("torch")
nn <- import("torch.nn", convert = TRUE)

resamplerCls_GAN <- function(x, y, lr = 0.01, batch_size = 32L, epochs = 100L, hidden_dim = 16L, noise_dim = 4L,
    n_needed = NULL) {
    Generator <- PyClass("Generator", inherit = nn$Module, defs = list(`__init__` = function(self, noise_dim,
        hidden_dim, output_dim) {
        super()$`__init__`()
        self$fc1 <- nn$Linear(noise_dim, hidden_dim)
        self$relu <- nn$ReLU()
        self$fc2 <- nn$Linear(hidden_dim, output_dim)
        NULL
    }, forward = function(self, z) {
        z <- self$relu(self$fc1(z))
        x_gen <- self$fc2(z)
        return(x_gen)
    }))

    Discriminator <- PyClass("Discriminator", inherit = nn$Module, defs = list(`__init__` = function(self,
        input_dim, hidden_dim) {
        super()$`__init__`()
        self$fc1 <- nn$Linear(input_dim, hidden_dim)
        self$relu <- nn$ReLU()
        self$fc2 <- nn$Linear(hidden_dim, 1L)
        self$sigmoid <- nn$Sigmoid()
        NULL
    }, forward = function(self, x) {
        x <- self$relu(self$fc1(x))
        x <- self$sigmoid(self$fc2(x))
        return(x)
    }))

    train_gan <- function(x_real_tensor, generator, discriminator, optim_G, optim_D, noise_dim, epochs,
        batch_size) {
        BCE <- nn$BCELoss()
        n <- x_real_tensor$size(0L)
        for (epoch in 1:epochs) {
            idx <- torch$randperm(n)
            for (i in seq(1, n, by = batch_size)) {
                end <- min(i + batch_size - 1, n)
                real_batch <- x_real_tensor[idx[i:end], ]
                b_size <- real_batch$size(0L)

                # Discriminator
                z <- torch$randn(b_size, noise_dim)
                fake_batch <- generator(z)$detach()
                real_labels <- torch$ones(c(b_size, 1L))
                fake_labels <- torch$zeros(c(b_size, 1L))

                optim_D$zero_grad()
                out_real <- discriminator(real_batch)
                out_fake <- discriminator(fake_batch)
                loss_D <- BCE(out_real, real_labels) + BCE(out_fake, fake_labels)
                loss_D$backward()
                optim_D$step()

                # Generator
                z <- torch$randn(b_size, noise_dim)
                fake_batch <- generator(z)
                optim_G$zero_grad()
                out_fake <- discriminator(fake_batch)
                loss_G <- BCE(out_fake, real_labels)
                loss_G$backward()
                optim_G$step()
            }
        }
    }

    generate_gan_samples <- function(generator, n_samples, noise_dim) {
        z <- torch$randn(n_samples, noise_dim)
        x_gen <- generator(z)
        return(x_gen)
    }

    ### --- Begin GAN Oversampling ---

    x <- as.matrix(x)
    p <- ncol(x)
    class_names <- levels(y)
    n_classes <- sapply(class_names, function(cl) sum(y == cl))
    k_class <- length(class_names)

    if (is.null(n_needed)) {
        n_needed <- max(n_classes) - n_classes
    }

    x_syn_list <- lapply(1:k_class, function(i) matrix(NA, nrow = 0, ncol = p))
    y_syn_list <- lapply(1:k_class, function(i) factor(c(), levels = levels(y)))

    for (i_class in 1:k_class) {
        if (n_needed[i_class] == 0)
            next

        x_main <- x[y == class_names[i_class], , drop = FALSE]
        x_main_tensor <- torch$tensor(x_main, dtype = torch$float32)

        generator <- Generator(noise_dim, hidden_dim, p)
        discriminator <- Discriminator(p, hidden_dim)

        optim_G <- torch$optim$Adam(generator$parameters(), lr = lr)
        optim_D <- torch$optim$Adam(discriminator$parameters(), lr = lr)

        train_gan(x_real_tensor = x_main_tensor, generator = generator, discriminator = discriminator, optim_G = optim_G,
            optim_D = optim_D, noise_dim = noise_dim, epochs = epochs, batch_size = batch_size)

        x_syn_temp <- generate_gan_samples(generator, n_samples = n_needed[i_class], noise_dim = noise_dim)
        x_syn_list[[i_class]] <- x_syn_temp$detach()$numpy()
        y_syn_list[[i_class]] <- factor(rep(class_names[i_class], n_needed[i_class]), levels = levels(y))
    }

    x_syn <- do.call(rbind, x_syn_list)
    y_syn <- do.call(c, y_syn_list)

    x_new <- rbind(x, x_syn)
    y_new <- c(y, y_syn)

    return(list(x_new = x_new, y_new = y_new, x_syn = x_syn, y_syn = y_syn))
}

dat <- datasets_classification_binary$banana
x <- dat[, 1:2]
y <- dat[, 3]

m_deepSMOTE <- resamplerCls_GAN(x = x, y = y, epochs = 200L, noise_dim = 10L)


plot(x, col = y)
points(x = m_deepSMOTE$x_syn, col = "green")
