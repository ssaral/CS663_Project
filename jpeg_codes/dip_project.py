import heapq
from collections import defaultdict, Counter
import numpy as np
import cv2
from scipy.fftpack import dct, idct

QY = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def rgb_to_ycbcr(image):
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    return ycbcr

def split_into_blocks(image, block_size=8):
    h, w = image.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            blocks.append(image[i:i+block_size, j:j+block_size])
    return blocks

def apply_dct(blocks):
    return [dct(dct(block.T, norm='ortho').T, norm='ortho') for block in blocks]

def quantize(blocks, quant_matrix):
    return [np.round(block / quant_matrix).astype(np.int32) for block in blocks]

def dequantize(blocks, quant_matrix):
    return [block * quant_matrix for block in blocks]

def apply_idct(blocks):
    return [idct(idct(block.T, norm='ortho').T, norm='ortho') for block in blocks]

def merge_blocks(blocks, image_shape, block_size=8):
    h, w = image_shape
    image = np.zeros((h, w))
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            image[i:i+block_size, j:j+block_size] = blocks[idx]
            idx += 1
    return image

    import heapq
from collections import defaultdict, Counter

class HuffmanNode:
    def __init__(self, value=None, frequency=0, left=None, right=None):
        self.value = value
        self.frequency = frequency
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.frequency < other.frequency

def build_huffman_tree(frequency_table):
    heap = [HuffmanNode(value, freq) for value, freq in frequency_table.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(frequency=left.frequency + right.frequency, left=left, right=right)
        heapq.heappush(heap, merged)
    
    return heap[0]

def generate_huffman_codes(node, prefix="", codebook={}):
    if node.value is not None:
        codebook[node.value] = prefix
    else:
        generate_huffman_codes(node.left, prefix + "0", codebook)
        generate_huffman_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_encode(blocks):
    flat_blocks = [value for block in blocks for value in block.flatten()]
    frequency_table = Counter(flat_blocks)
    
    root = build_huffman_tree(frequency_table)
    codebook = generate_huffman_codes(root)
    
    encoded_data = "".join(codebook[value] for value in flat_blocks)
    return encoded_data, codebook


def huffman_decode(encoded_data, codebook, original_shape, block_size=8):
    reverse_codebook = {code: value for value, code in codebook.items()}
    
    decoded_values = []
    buffer = ""
    for bit in encoded_data:
        buffer += bit
        if buffer in reverse_codebook:
            decoded_values.append(reverse_codebook[buffer])
            buffer = ""
    
    h, w = original_shape
    num_blocks = (h // block_size) * (w // block_size)
    blocks = []
    idx = 0
    for _ in range(num_blocks):
        block = np.array(decoded_values[idx:idx + block_size**2]).reshape((block_size, block_size))
        blocks.append(block)
        idx += block_size**2
    return blocks

def rle_encode(blocks):
    encoded_blocks = []
    for block in blocks:
        flat_block = block.flatten()
        rle_block = []
        count = 1
        for i in range(1, len(flat_block)):
            if flat_block[i] == flat_block[i - 1]:
                count += 1
            else:
                rle_block.append((flat_block[i - 1], count))
                count = 1
        rle_block.append((flat_block[-1], count))
        encoded_blocks.append(rle_block)
    return encoded_blocks

def rle_decode(encoded_blocks, block_size=8):
    decoded_blocks = []
    for rle_block in encoded_blocks:
        decoded_flat_block = []
        for value, count in rle_block:
            decoded_flat_block.extend([value] * count)
        decoded_blocks.append(np.array(decoded_flat_block).reshape((block_size, block_size)))
    return decoded_blocks

def jpeg_compression_grayscale(image, quality_factor=50):
    scale = 50 / quality_factor if quality_factor < 50 else 2 - quality_factor / 50
    qy = np.clip(QY * scale, 1, 255).astype(np.int32)

    blocks = split_into_blocks(image)

    dct_blocks = apply_dct(blocks)
    quantized_blocks = quantize(dct_blocks, qy)

    return quantized_blocks, qy, image.shape

def jpeg_decompression_grayscale(quantized_blocks, quant_matrix, original_shape):
    dequantized_blocks = dequantize(quantized_blocks, quant_matrix)
    idct_blocks = apply_idct(dequantized_blocks)

    decompressed_image = merge_blocks(idct_blocks, original_shape)
    
    return np.clip(decompressed_image, 0, 255).astype(np.uint8)

image = cv2.imread('kodim02.png', cv2.IMREAD_GRAYSCALE)

quantized_blocks, quant_matrix, original_shape = jpeg_compression_grayscale(image, quality_factor=50)

encoded_data, codebook = huffman_encode(quantized_blocks)

rle_encoded_blocks = rle_encode(quantized_blocks)

print(len(image))
print(len(rle_encoded_blocks))

rle_decoded_blocks = rle_decode(rle_encoded_blocks)

decompressed_image = jpeg_decompression_grayscale(quantized_blocks, quant_matrix, original_shape)

cv2.imwrite('compressed_grayscale_image.jpg', decompressed_image)

cv2.imshow('Original Grayscale Image', image)
cv2.imshow('Compressed Grayscale Image', decompressed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

