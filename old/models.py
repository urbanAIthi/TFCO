class SpacialDecoder(nn.Module):
    def __init__(self):
        super(SpacialDecoder, self).__init__()
        self.cnn_Autoencoder = Autoencoder()

        #Initialize the ViT encoder
        self.vit_encoder = ViT(
                            image_size = 400,
                            patch_size = 50,
                            num_classes = 1000,
                            dim = 1024,
                            depth = 6,
                            heads = 8,
                            mlp_dim = 1024,
                            dropout = 0.25,
                            emb_dropout = 0,
                            channels = 1
                        )
        
        dim = 1024
        temporal_depth = 6
        heads = 8
        dim_head = 1024
        mlp_dim = 1024
        dropout = 0
        self.global_average_pool = 'cls'
        self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout)
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        s, b, c, h, w = img.shape
        img = img.reshape(s*b, c, h, w)
        #ViT encoder
        encoded = self.vit_encoder(img)
        x = encoded.reshape(s, b, -1)

        # append temporal CLS tokens

        if exists(self.temporal_cls_token):
            temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b = b)

            x = torch.cat((temporal_cls_tokens, x), dim = 1)

        # attend across time

        x = self.temporal_transformer(x)

        # excise out temporal cls token or average pool

        x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

        x = self.cnn_Autoencoder.fc2(encoded)
        encoded_fc = self.cnn_Autoencoder.fc_relu2(x)

        # Reshape the encoded tensor
        encoded_fc_reshaped = encoded_fc.view(-1, 16, 50, 50)

        # Decoder
        x = self.cnn_Autoencoder.decoder_conv1(encoded_fc_reshaped)
        x = self.cnn_Autoencoder.decoder_relu1(x)
        x = self.cnn_Autoencoder.decoder_conv2(x)
        x = self.cnn_Autoencoder.decoder_relu2(x)
        x = self.cnn_Autoencoder.decoder_conv3(x)
        decoded = self.cnn_Autoencoder.decoder_sigmoid(x)
        #print(f'cnn_decoder time: {time.time() - t}')
        return decoded

class SpacialTemporalDecoderOld(nn.Module):
    def __init__(self, config, fix_decoder: bool = False, load_decoder: Union[None, str] = None,
                 fix_spacial: bool = False, load_spacial: Union[None, str] = None):
        super(SpacialTemporalDecoder, self).__init__()
        #Initialize pure cnn autoencoder model
        self.cnn_Autoencoder = Autoencoder()

        """ #Load the pretrained pure autoencoder model
        if load_decoder is not None:
            state_dict = torch.load(load_decoder)
            desired_keys = [key for key in state_dict.keys() if key.startswith('cnn_Autoencoder.decoder')]
            filtered_state_dict = {key: state_dict[key] for key in desired_keys}
            filtered_state_dict = {key.replace('cnn_Autoencoder.', ''): filtered_state_dict[key] for key in desired_keys}
            self.cnn_Autoencoder.load_state_dict(filtered_state_dict, strict=False)
        if fix_decoder:
            for param in self.cnn_Autoencoder.parameters():
                param.requires_grad = False """

        #Initialiaze the Temporal Transformer
        self.temporal_transformer = TemporalTransformer()

        #Initialize the ViT encoder
        self.vit_encoder = ViT(
                            image_size = 400,
                            patch_size = 50,
                            num_classes = 1000,
                            dim = 1024,
                            depth = 6,
                            heads = 8,
                            mlp_dim = 1024,
                            dropout = 0.25,
                            emb_dropout = 0,
                            channels = 1
                        )


        """ if load_decoder is not None:
            #load the pretrained vit_encoder_cnn_decoder model
            state_dict = torch.load(load_decoder)
            #extract the vit_encoder state_dict
            desired_keys = [key for key in state_dict.keys() if key.startswith('vit_encoder')]
            filtered_state_dict = {key: state_dict[key] for key in desired_keys}
            filtered_state_dict = {key.replace('vit_encoder.', ''): filtered_state_dict[key] for key in desired_keys}
            # print(filtered_state_dict.keys())
            self.vit_encoder.load_state_dict(filtered_state_dict)
        if fix_spacial:
            for param in self.vit_encoder.parameters():
                param.requires_grad = False """
                

        # create randon torch tensor with shape (sequence_len = 10, batch_size = 32, num_channels = 1, height = 400, width = 400)
        # t = torch.rand(10, 32, 1, 400, 400)
        # reshape to (sequence_len * batch_size, num_channels, height, width)
        # t = t.reshape(320, 1, 400, 400)
        # pass tensor through the special (vit) encoder
        # model = Autoencoder()
        # t = torch.rand(10, 32, 1024)
        # pass through temporal decoder
        # t = torch.rand(32, 1024)
        # pass through cnn decoder to reconstruct image
        # t = torch.rand(32, 1, 400, 400)

    def forward(self, img_sequence):
        t = time.time()
        s, b, c, h, w = img_sequence.size()
        concat_sequence = img_sequence.reshape(s*b, c, h, w)
        #ViT encoder
        x = self.vit_encoder(concat_sequence)
        #print(f'vit_encoder time: {time.time() - t}')

        encoded_sequene = x.reshape(s, b, -1)
        #print(f'encoded_sequene shape: {encoded_sequene.shape}')

        #Temporal Transformer
        temporal_transformed = self.temporal_transformer(encoded_sequene)
        #print(f'temporal_transformed shape: {temporal_transformed.shape}')
        #print(f'temporal_transformer time: {time.time() - t}')
        #CNN decoder

        x = self.cnn_Autoencoder.fc2(temporal_transformed)
        encoded_fc = self.cnn_Autoencoder.fc_relu2(x)

        # Reshape the encoded tensor
        encoded_fc_reshaped = encoded_fc.view(-1, 16, 50, 50)

        # Decoder
        x = self.cnn_Autoencoder.decoder_conv1(encoded_fc_reshaped)
        x = self.cnn_Autoencoder.decoder_relu1(x)
        x = self.cnn_Autoencoder.decoder_conv2(x)
        x = self.cnn_Autoencoder.decoder_relu2(x)
        x = self.cnn_Autoencoder.decoder_conv3(x)
        decoded = self.cnn_Autoencoder.decoder_sigmoid(x)
        #print(f'cnn_decoder time: {time.time() - t}')
        return decoded


class ViViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        spatial_depth,
        temporal_depth,
        heads,
        mlp_dim,
        pool = 'cls',
        channels = 1,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_image_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = (frames // frame_patch_size)

        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.global_average_pool = pool == 'mean'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )


        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None

        self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)
        self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

        self.cnn_Autoencoder = Autoencoder()
        # Load the pretrained pure autoencoder model
        self.cnn_Autoencoder.load_state_dict(torch.load('models/newvectoram_spacial_decoder.pth'), strict=False)
        self.cnn_head = nn.Sequential(
            self.cnn_Autoencoder.fc2,
            self.cnn_Autoencoder.fc_relu2,
            Reshape(),
            self.cnn_Autoencoder.decoder_conv1,
            self.cnn_Autoencoder.decoder_relu1,
            self.cnn_Autoencoder.decoder_conv2,
            self.cnn_Autoencoder.decoder_relu2,
            self.cnn_Autoencoder.decoder_conv3,
            self.cnn_Autoencoder.decoder_sigmoid
        )

        self.cnn_encoder = CNNEncoder()

        self.use_cnn_encoder = True
        self.use_vit_encoder = False
        assert not (self.use_cnn_encoder and self.use_vit_encoder), 'Only one encoder can be used at a time'


    def forward(self, video):
        if self.use_vit_encoder:
            s, b, c, h, w = video.shape
            video = video.reshape(b,c,s,h,w)
            x = self.to_patch_embedding(video)
            b, f, n, _ = x.shape

            x = x + self.pos_embedding[:, :f, :n]

            if exists(self.spatial_cls_token):
                spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b = b, f = f)
                x = torch.cat((spatial_cls_tokens, x), dim = 2)

            x = self.dropout(x)

            x = rearrange(x, 'b f n d -> (b f) n d')

            # attend across space

            x = self.spatial_transformer(x)

            x = rearrange(x, '(b f) n d -> b f n d', b = b)

            # excise out the spatial cls tokens or average pool for temporal attention

            x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')
        
        elif self.use_cnn_encoder:
            s,b,c, h, w = video.shape
            video = video.reshape(b,s,c,h,w)
            x = video.reshape(b*s,c,h,w)
            x = self.cnn_encoder(x)
            x = x.reshape(b,s,-1)
            

        '''

        # append temporal CLS tokens

        if exists(self.temporal_cls_token):
            temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b = b)

            x = torch.cat((temporal_cls_tokens, x), dim = 1)

        # attend across time

        x = self.temporal_transformer(x)

        # excise out temporal cls token or average pool

        x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

        '''
        x = torch.squeeze(x, dim=1)
        x = self.to_latent(x)
        return self.cnn_head(x)


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # The final fully connected layer will take the flattened vector from the last convolutional layer,
        # and output a vector of size 1024
        self.fc = nn.Linear(in_features=128*25*25, out_features=1024)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten the output from the convolutional layers
        x = x.view(-1, 128*25*25)
        
        # Pass through the fully connected layer
        x = self.fc(x)
        
        return x