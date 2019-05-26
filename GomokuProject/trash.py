


layer1 = slim.conv2d(input_dict["obs"],8,3)
        layer2 = slim.conv2d(layer1,8,3)
        layer3 = slim.flatten(layer2)
        layer4 = slim.fully_connected(layer3, 100)
        return layer4,layer3