#imports

import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy.signal import fftconvolve

### EQUALIZER DESIGN IS HERE

def autocorr_mtx(channel, noise_variance, equalizer_length):
    """
    Generate an autocorrelation matrix for a given channel estimate, padded for longer equalizers.

    Parameters:
        channel (array-like): Channel coefficients [h0, h1, ..., hn].
        noise_variance (float): Noise variance to be included in the diagonal entries.
        equalizer_length (int): Length of the desired equalizer (must be >= channel length).

    Returns:
        np.ndarray: The autocorrelation matrix R.
    """
    h = np.array(channel, dtype=complex)
    num_taps = len(h)

    if equalizer_length < num_taps:
        raise ValueError("Equalizer length must be greater than or equal to the channel length.")

    # Initialize an empty autocorrelation matrix
    R = np.zeros((equalizer_length, equalizer_length), dtype=complex)

    for i in range(equalizer_length):
        for j in range(equalizer_length):
            shift = abs(i - j)
            if shift < num_taps:
                # Compute the shifted inner product
                overlap_length = num_taps - shift
                # R[i, j] = np.dot(h[:overlap_length], h[shift:shift + overlap_length])
                R[i, j] = np.dot(h[:overlap_length], np.conj(h[shift:shift + overlap_length]))
            else:
                # Out-of-bounds indices contribute zero
                R[i, j] = 0

            if i == j:
                # Add noise variance to the diagonal
                R[i, j] += noise_variance

    return R



def autocorr_mtx_old(channel, noise_variance, equalizer_length):
    """
    Generate an autocorrelation matrix for a given channel estimate, padded for longer equalizers.

    Parameters:
        channel_estimates (array-like): Channel coefficients [h0, h1, ..., hn].
        noise_variance (float): Noise variance to be included in the diagonal entries.
        equalizer_length (int): Length of the desired equalizer (must be >= channel length).

    Returns:
        np.ndarray: The autocorrelation matrix R.
    """
    h = np.array(channel, dtype=complex)
    num_taps = len(h)

    if equalizer_length < num_taps:
        raise ValueError("Equalizer length must be greater than or equal to the channel length.")

    # Initialize an empty autocorrelation matrix
    R = np.zeros((equalizer_length, equalizer_length), dtype=complex)

    for i in range(equalizer_length):
        for j in range(equalizer_length):
            shift = abs(i - j)
            if shift < num_taps:
                # Compute the shifted inner product
                # R[i, j] = np.sum(h[:num_taps-shift] * np.conj(h[shift:num_taps]))
                R[i, j] = np.sum(h[:num_taps-shift] * h[shift:num_taps])
            else:
                # Out-of-bounds indices contribute zero
                R[i, j] = 0

            if i == j:
                # Add noise variance to the diagonal
                R[i, j] += noise_variance

    return R

def corr_vector(channel, equalizer_length):

  # TODO: Improve robustness

    """
    Generate a correlation vector for a given channel and equalizer length.
    In current implementation, works for unit energy tx constellations.

    Parameters:
        channel (array-like): Channel coefficients [h0, h1, ..., hn].
        equalizer_length (int): Length of the desired equalizer.

    Returns:
        np.ndarray: The correlation vector p.
    """
    h = np.array(channel, dtype=complex)
    num_taps = len(h)

    if equalizer_length < num_taps:
        raise ValueError("Equalizer length must be greater than or equal to the channel length.")

    # Initialize the correlation vector with zeros
    p = np.zeros(equalizer_length, dtype=complex)

    # Populate the vector based on the channel coefficients
    for i in range(equalizer_length):
        # if i < num_taps:
        #     p[i] = np.conj(h[i])  # Include conjugate of the channel tap
        if i == 0:
          p[i] = h[i]
        else:
          p[i] = 0  # Zero padding for extended equalizer length

    return p


def lmmse_weights(channel, noise_variance, equalizer_length):
  R = autocorr_mtx(channel, noise_variance, equalizer_length)
  p = corr_vector(channel, equalizer_length)
  # print(f'R: {R}')
  # print(f'p: {p}')
  return np.linalg.inv(R) @ p


def equalize(rx, equalizer_weights):
  length = len(rx)
  eq = np.convolve(rx, equalizer_weights)
  return eq[:length]

### PILOT TONE, CHANNEL ESTIMATION, AND NOISE HERE

def add_pilots(tx, num_pilots, channel_length):
  """
  Adds pilot tones to tx. The pilots are spaced according to the
  channel length/delay spread. (let P be the pilot symbol: P000...00)

  Parameters:
  tx: the message the pilot symbols will be added to
  num_pilots: the number of pilot symbols added to the
  front of the transmission. more pilots --> more accurate estimate
  channel_length: number of taps for channel impulse response, digital "delay spread"

  Returns:
  tx_pilot: the message with pilot symbols added to the front
  """
  tx_pilot = tx
  for i in range(num_pilots):
    tx_pilot = np.insert(tx_pilot, 0, np.zeros(channel_length - 1))
    tx_pilot = np.insert(tx_pilot, 0 , 1)
  return tx_pilot


def channel_estimate(rx, num_pilots, channel_length):
    """
    Estimates a channel from pilots. The pilots must be spaced
    as in the "add_pilots" function. (One tx of "1" followed by
    zeros for the remaining delay spread)

    Parameters:
        rx: the raw received digital transmission
        num_pilots: the number of pilot symbols added to the front of the transmission
        channel_length: number of taps for channel impulse response, digital "delay spread"

    Returns:
        h_hat: average of channel estimate from each pilot
    """
    estimates = np.zeros((num_pilots, channel_length), dtype=complex)

    # Iterate through each pilot to estimate the channel
    for i in range(num_pilots):
        # Extract the corresponding received segment for this pilot
        rx_segment = rx[i * channel_length:(i + 1) * channel_length]

        # Use the known pilot symbol (1) to estimate the channel
        if len(rx_segment) == channel_length:
            estimates[i, :] = rx_segment / (1)

    # Average the estimates across all pilots to get the final channel estimate
    h_hat = np.mean(estimates, axis=0)

    return h_hat


def remove_pilots(rx, num_pilots, channel_length):
  return rx[channel_length*num_pilots:]


def add_noise(tx, noise_variance):
  return tx + np.sqrt(noise_variance/2)* np.random.randn(len(tx), 2).view(np.complex128)


### BITSTREAM GENERATION AND DECODING

binary = np.array([0, 1])

def generate_bitstream(length):
  return np.array([random.choice(binary) for _ in range(length)])

def binary_to_qpsk(bits):
  bits = np.array(bits, dtype=int)

  if len(bits) % 2 != 0:
    raise ValueError('Input binary length must be even.')

  qpsk_pairs = bits.reshape(-1, 2)

  qpsk_map = {
      (1,1): 1+1j,
      (1, 0): 1-1j,
      (0, 1): -1+1j,
      (0, 0): -1-1j
  }

  qpsk_data = np.array([qpsk_map[tuple(pair)] for pair in qpsk_pairs])
  return qpsk_data /np.sqrt(2)

def binary_to_bpsk(bits):
  bits = np.array(bits, dtype=int)
  return 2*bits - 1

def qpsk_to_binary(qpsk_data):
  qpsk_data = np.array(qpsk_data, dtype=complex)

  qpsk_map = {
      1+1j: (1,1),
      1-1j: (1,0),
      -1+1j: (0,1),
      -1-1j: (0,0)
  }

  binary_pairs = np.array([qpsk_map[data] for data in qpsk_data])
  return binary_pairs.flatten()


def bpsk_to_binary(bpsk_data):
  bpsk_data = np.array(bpsk_data, dtype=complex)
  return (bpsk_data + 1) / 2


### ---- ABOVE FUNCTIONS BEGIN WITH BINARY, BELOW WITH PURE SYMBOLS -----


qpsk_constellation = 1/np.sqrt(2) * np.array([-1-1j, -1+1j,  1-1j, 1+1j])
bpsk_constellation = np.array([-1, 1])

def generate_qpsk(length):
  return np.array([random.choice(qpsk_constellation) for _ in range(length)])

def generate_bpsk(length):
  return np.array([random.choice(bpsk_constellation) for _ in range(length)])

def qpsk_decoder(rx):
  # decodes QPSK transmissions
  output_real = np.sign(rx.real)
  output_imag = np.sign(rx.imag)
  return (output_real + 1j*output_imag)

def bpsk_decoder(rx):
  # decodes BPSK transmissions
  return np.sign(rx.real)

def error_calc(tx, rx):
  #calculates error rate
  return np.sum(tx != rx)/len(tx)


#CHANNEL SIMULATION

def channel_sim(tx, channel, noise_variance = 0):
  full_rx = np.convolve(tx, channel, 'full')
  trunc_rx = full_rx[0:len(tx)]
  noise = np.sqrt(noise_variance/2)* np.random.randn(len(trunc_rx), 2).view(np.complex128)
  return trunc_rx + noise.flatten()



# Generator and Parity Check

def hamming_encode(msg):
  """
    Encode a 4-bit message using the (7,4) Hamming code.

    Parameters:
        msg (list or array): A 4-bit binary message (e.g., [1, 0, 0, 1]).

    Returns:
        np.ndarray: A 7-bit encoded codeword.
    """
    # Generator matrix for (7,4) Hamming code

  G = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ])

  msg = np.array(msg, dtype=int)
  return np.mod(G @ msg, 2)

def hamming_decode(rx, ecc = True):
  """
    Decode and correct a received 7-bit codeword using the (7,4) Hamming code.

    Parameters:
        rx (list or array): A 7-bit binary codeword (e.g., [1, 0, 0, 1, 1, 0, 0]).

    Returns:
        tuple: decoded_message
            - decoded_message (np.ndarray): The corrected 4-bit message.
    """
    # Parity-check matrix for (7,4) Hamming code
  H = np.array([
        [1, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1],
        [0, 0, 1, 1, 1, 0, 1]
      ])

  rx = np.array(rx, dtype=int)

  #caluclate error vector
  syndrome = np.mod(rx @ np.transpose(H), 2)
  # print(f'syndrome vector: {syndrome}')

  if not ecc:
    return rx[3:]

  if np.count_nonzero(syndrome) == 0:
    return rx[3:]  # No error, return the first 4 bits

  syndrome_map = {4:1, 2:2, 1:3, 3:4, 5:5, 6:6, 7:7, 0:0}

  syndrome_decimal = syndrome_map[int("".join(map(str, syndrome)), 2)] - 1  # Convert syndrome to 0-based index


  if 0 < syndrome_decimal < len(rx):
        # print(f'flipped bit: {rx[syndrome_decimal]} at position {syndrome_decimal}')
        rx[syndrome_decimal] ^= 1  # Flip the erroneous bit

    # Return the corrected message (first 4 bits) and error status
  return rx[3:]

# Split longer messages

def ecc_74_encode(msg):
  msg = np.array(msg, dtype=int)

  if len(msg) % 4 != 0:
    raise ValueError('Input message length must be a multiple of 4.')

  ecc_blocks = msg.reshape(-1, 4)
  encoded_blocks = np.array([hamming_encode(block) for block in ecc_blocks])
  return encoded_blocks.flatten()

def ecc_74_decode(rx, ecc = False):
  rx = np.array(rx, dtype=int)
  if len(rx) % 7 != 0:
    raise ValueError('Input message length must be a multiple of 7.')

  # Reshape to process blocks of 7
  ecc_blocks = rx.reshape(-1, 7)

  # Apply hamming_decode and ensure a consistent output shape
  decoded_blocks = [hamming_decode(block, ecc) for block in ecc_blocks]
  decoded_blocks = np.vstack(decoded_blocks) #added this to stack results, may need to change to hstack depending on if input is column or row major

  return decoded_blocks.flatten()

# New Idea: Seperate classes for different parts of simulation process:
"""
Classes:
 - Equalizer Class
 - Channel Class
 - Modulator/Demodulator Class
"""

# EXTREMELY NOT DONE


class RadioModel:
  def __init__(self, modulation, channel_coefficients, noise_variance):
    # Initialize sub-classes
    self.modulator = Modulator(modulation)
    self.channel = Channel(channel_coefficients, noise_variance)
    self.equalizer = Equalizer(channel_coefficients)

    # Centralized parameters
    self.bit_length = 2000
    self.tx_signal = None
    self.rx_signal = None
    self.equalized_signal = None

  def summary(self):
    print("Radio Model Summary:")
    print('')
    print('--------------------------------')
    print('--------------------------------')
    print('')
    self.modulator.summary()
    self.channel.summary()
    self.equalizer.summary()
    print('--------------------------------')


class Equalizer(RadioModel):
  '''
  Performs channel estimation and equlization on recieved signals.
  Currently supports ZF and LMMSE, plans to include decision feedback and SIC later
  '''
  def __init__(self, method = 'zf', channel = None, noise_var = 0, eq_length = None, weights = None):
    self.channel = channel
    self.noise_var = noise_var
    self.method = method
    self.weights = weights

    if not eq_length:
      self.eq_length = len(channel) if channel else 0
    else:
      self.eq_length = eq_length

  def summary(self):
    print('Equalizer Summary')
    print('--------------------------------')
    print(f'Channel Estimate: {self.channel}')
    print(f'Noise Variance: {self.noise_var}')
    print(f'Equalizer Length: {self.eq_length}')
    print(f'Equalizer Mode: {self.method}')
    print('--------------------------------')

  def channel_estimate(self, rx, num_pilots, eq_length):
    h_hat = channel_estimate(rx, num_pilots, eq_length)
    self.h_hat = h_hat
    return h_hat

  def set_method(self, method):
    self.method = method

  def set_channel(self, channel, noise_variance = 0):
    self.channel = channel
    self.noise_var = noise_variance

  def compute_weights(self):
    if self.method == 'zf':
      w = lmmse_weights(self.channel, 0, self.eq_length)
    elif self.method == 'lmmse':
      w = lmmse_weights(self.channel, self.noise_var, self.eq_length)
    else:
      raise ValueError('Invalid equalizer method. Must be "zf" or "lmmse".')

    self.weights = w
    return w

  def equalize(self, rx):
    eq = equalize(rx, self.weights)
    return eq





class Modulator(RadioModel):
  def __init__(self, modulation, ecc_method = None, num_pilots = 0, pilot_spacing = 1):
    '''
    Class handles generation, modulation, pilot encoding, and error correction
    Currently supports BPSK, QPSK
    Pilot encoding done very naively, will improve once I'm smarter
    Only Hamming rate 7/4 supported right now. Other rates as well as other
    ECC methods later
    '''
    self.modulation = modulation
    self.ecc_method = ecc_method
    self.num_pilots = num_pilots
    self.pilot_spacing = pilot_spacing

    self.binary_msg = None
    self.mod_msg = None
    self.ecc_msg = None

    self.rx = None
    self.rx_pilots_removed = None
    self.rx_demod = None
    self.rx_ecc = None

  def summary(self):
    print('Modulator Summary')
    print('--------------------------------')
    print(f'Modulation: {self.modulation}')
    print(f'Error Correction Scheme: {self.ecc_method}')
    print(f'Num Pilots: {self.num_pilots}')
    print(f'Pilot Spacing: {self.pilot_spacing}')
    print('--------------------------------')
    print('')

  def set_modulation(self, modulation):
    self.modulation = modulation

  def set_ecc(self, method):
    self.ecc_method = method

  def set_num_pilots(self, num_pilots):
    self.num_pilots = num_pilots

  def num_pilots(self):
    return self.num_pilots

  def eq_length(self):
    return self.eq_length

  def set_pilot_spacing(self, pilot_spacing):
    # Pilot spacing depends on channel length.
    self.pilot_spacing = pilot_spacing

  def generate_binary(self, length):
    self.binary_msg = np.array([random.choice(binary) for _ in range(length)])
    return self.binary_msg

  def ecc_encode(self, msg = None):
    if not msg:
      msg = self.binary_msg

    # Rebuild this as a Command Pattern as ECC capibilities grow
    if self.ecc_method == 'ham74':
      self.ecc_msg = ecc_74_encode(msg)
    else:
      raise ValueError('Invalid ECC method. Must be "ham74" (for now...)')

  def modulate(self, msg = None):
    if msg is None:
        msg = self.ecc_msg or self.binary_msg # If both are declared, python returns left object
        if msg is None:
            raise ValueError("No message provided or available for modulation.")

    # Rebuild this as a Command Pattern as modulation capibilities grow
    if self.modulation == 'qpsk':
      self.mod_msg = binary_to_qpsk(msg)
    elif self.modulation == 'bpsk':
      self.mod_msg = binary_to_bpsk(msg)

  def add_pilots(self, tx):
    # Pilot spacing is determined by channel length
    self.tx_pilot = add_pilots(tx, self.num_pilots, self.pilot_spacing)
    return self.tx_pilot

  def tx_pipeline(self, binary_msg = None):
    if not binary_msg:
      binary_msg = self.binary_msg

    self.ecc_encode(binary_msg)
    self.modulate()
    self.add_pilots(self.mod_msg)
    return self.tx_pilot

  ### ---- TX FUNCTIONS ABOVE, RX PROCESSING BELOW

  def remove_pilots(self, eq_msg):
    self.rx_pilots_removed = remove_pilots(eq_msg, self.num_pilots, self.pilot_spacing)

  def demodulate(self, eq_msg = None):
    if not eq_msg:
      eq_msg = self.rx_pilots_removed

    if self.modulation == 'qpsk':
      qpsk_demod = qpsk_decoder(eq_msg)
      self.rx_demod = qpsk_to_binary(qpsk_demod)
    elif self.modulation == 'bpsk':
      bpsk_demod = bpsk_decoder(eq_msg)
      self.rx_demod = bpsk_to_binary(bpsk_demod)

  def ecc_decode(self, rx_binary = None):
    if not rx_binary:
      rx_binary = self.rx_demod

    if self.ecc_method == 'ham74':
      self.rx_ecc = ecc_74_decode(rx_binary)
      return self.rx_ecc
    else:
      raise ValueError('Invalid ECC method. Must be "ham74" (for now...)')


  def rx_pipeline(self, eq_msg):
    self.remove_pilots(eq_msg)
    self.demodulate()
    self.ecc_decode()



class Channel(RadioModel):
  '''
  Channel model for basic SISO transmission.
  Incoming features: MIMO, OFDM, better simulation tools
  '''
  def __init__(self, h, noise_var, tx_gain = 1):
    self.h = h
    self.noise_var = noise_var
    self.tx_gain = tx_gain

  def summary(self):
    print('Channel Summary')
    print('--------------------------------')
    print(f'Channel: {self.h}')
    print(f'Noise Variance: {self.noise_var}')
    print(f'Transmit Gain: {self.tx_gain}')
    print('--------------------------------')

  def set_channel(self, channel, noise_variance = 0):
    self.h = channel
    self.noise_var = noise_variance

  @property
  def length(self):
    return len(self.h)

  def channel_sim(self, tx):
    self.rx = channel_sim(self.tx_gain * tx, self.h, self.noise_var)
    return self.rx