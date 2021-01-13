import numpy as np
import ase
import matplotlib.pyplot as p
from numpy.linalg import norm, svd
import soaputils as su
import time

import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import sys
import scipy.optimize as so
from ase.io import read, write

'''
This code computes the Bayesian Optimization for given boundary conditions of a structure for a given error function using the GPyOpt implementation.
As inputs the scale factor has to be specified, and a factor to specify how much iterations should be performed.
'''

# input stuff:
n = sys.argv[1] # just an index
s1 = sys.argv[2] # scalefactor
s1=float(s1)
print(s1)

Natoms = 150
atoms = su.gen_struct(Natoms, seed=50, elements=['Cu'])

def black_box_function(x):
    """
    Define the error function to minimize.
    Unfortunately you have to name every degree of freedom personaly, which is a bit of annoying and lenghty.
    """
    x1 = x[:,0]
    x2 = x[:,1]
    x3 = x[:,2]
    x4 = x[:,3]
    x5 = x[:,4]
    x6 = x[:,5]
    x7 = x[:,6]
    x8 = x[:,7]
    x9 = x[:,8]
    x10 = x[:,9]
    x11 = x[:,10]
    x12 = x[:,11]
    x13 = x[:,12]
    x14 = x[:,13]
    x15 = x[:,14]
    x16 = x[:,15]
    x17 = x[:,16]
    x18 = x[:,17]
    x19 = x[:,18]
    x20 = x[:,19]
    x21 = x[:,20]
    x22 = x[:,21]
    x23 = x[:,22]
    x24 = x[:,23]
    x25 = x[:,24]
    x26 = x[:,25]
    x27 = x[:,26]
    x28 = x[:,27]
    x29 = x[:,28]
    x30 = x[:,29]
    x31 = x[:,30]
    x32 = x[:,31]
    x33 = x[:,32]
    x34 = x[:,33]
    x35 = x[:,34]
    x36 = x[:,35]
    x37 = x[:,36]
    x38 = x[:,37]
    x39 = x[:,38]
    x40 = x[:,39]
    x41 = x[:,40]
    x42 = x[:,41]
    x43 = x[:,42]
    x44 = x[:,43]
    x45 = x[:,44]
    x46 = x[:,45]
    x47 = x[:,46]
    x48 = x[:,47]
    x49 = x[:,48]
    x50 = x[:,49]
    x51 = x[:,50]
    x52 = x[:,51]
    x53 = x[:,52]
    x54 = x[:,53]
    x55 = x[:,54]
    x56 = x[:,55]
    x57 = x[:,56]
    x58 = x[:,57]
    x59 = x[:,58]
    x60 = x[:,59]
    x61 = x[:,60]
    x62 = x[:,61]
    x63 = x[:,62]
    x64 = x[:,63]
    x65 = x[:,64]
    x66 = x[:,65]
    x67 = x[:,66]
    x68 = x[:,67]
    x69 = x[:,68]
    x70 = x[:,69]
    x71 = x[:,70]
    x72 = x[:,71]
    x73 = x[:,72]
    x74 = x[:,73]
    x75 = x[:,74]
    x76 = x[:,75]
    x77 = x[:,76]
    x78 = x[:,77]
    x79 = x[:,78]
    x80 = x[:,79]
    x81 = x[:,80]
    x82 = x[:,81]
    x83 = x[:,82]
    x84 = x[:,83]
    x85 = x[:,84]
    x86 = x[:,85]
    x87 = x[:,86]
    x88 = x[:,87]
    x89 = x[:,88]
    x90 = x[:,89]
    x91 = x[:,90]
    x92 = x[:,91]
    x93 = x[:,92]
    x94 = x[:,93]
    x95 = x[:,94]
    x96 = x[:,95]
    x97 = x[:,96]
    x98 = x[:,97]
    x99 = x[:,98]
    x100 = x[:,99]
    x101 = x[:,100]
    x102 = x[:,101]
    x103 = x[:,102]
    x104 = x[:,103]
    x105 = x[:,104]
    x106 = x[:,105]
    x107 = x[:,106]
    x108 = x[:,107]
    x109 = x[:,108]
    x110 = x[:,109]
    x111 = x[:,110]
    x112 = x[:,111]
    x113 = x[:,112]
    x114 = x[:,113]
    x115 = x[:,114]
    x116 = x[:,115]
    x117 = x[:,116]
    x118 = x[:,117]
    x119 = x[:,118]
    x120 = x[:,119]
    x121 = x[:,120]
    x122 = x[:,121]
    x123 = x[:,122]
    x124 = x[:,123]
    x125 = x[:,124]
    x126 = x[:,125]
    x127 = x[:,126]
    x128 = x[:,127]
    x129 = x[:,128]
    x130 = x[:,129]
    x131 = x[:,130]
    x132 = x[:,131]
    x133 = x[:,132]
    x134 = x[:,133]
    x135 = x[:,134]
    x136 = x[:,135]
    x137 = x[:,136]
    x138 = x[:,137]
    x139 = x[:,138]
    x140 = x[:,139]
    x141 = x[:,140]
    x142 = x[:,141]
    x143 = x[:,142]
    x144 = x[:,143]
    x145 = x[:,144]
    x146 = x[:,145]
    x147 = x[:,146]
    x148 = x[:,147]
    x149 = x[:,148]
    x150 = x[:,149]
    x151 = x[:,150]
    x152 = x[:,151]
    x153 = x[:,152]
    x154 = x[:,153]
    x155 = x[:,154]
    x156 = x[:,155]
    x157 = x[:,156]
    x158 = x[:,157]
    x159 = x[:,158]
    x160 = x[:,159]
    x161 = x[:,160]
    x162 = x[:,161]
    x163 = x[:,162]
    x164 = x[:,163]
    x165 = x[:,164]
    x166 = x[:,165]
    x167 = x[:,166]
    x168 = x[:,167]
    x169 = x[:,168]
    x170 = x[:,169]
    x171 = x[:,170]
    x172 = x[:,171]
    x173 = x[:,172]
    x174 = x[:,173]
    x175 = x[:,174]
    x176 = x[:,175]
    x177 = x[:,176]
    x178 = x[:,177]
    x179 = x[:,178]
    x180 = x[:,179]
    x181 = x[:,180]
    x182 = x[:,181]
    x183 = x[:,182]
    x184 = x[:,183]
    x185 = x[:,184]
    x186 = x[:,185]
    x187 = x[:,186]
    x188 = x[:,187]
    x189 = x[:,188]
    x190 = x[:,189]
    x191 = x[:,190]
    x192 = x[:,191]
    x193 = x[:,192]
    x194 = x[:,193]
    x195 = x[:,194]
    x196 = x[:,195]
    x197 = x[:,196]
    x198 = x[:,197]
    x199 = x[:,198]
    x200 = x[:,199]
    x201 = x[:,200]
    x202 = x[:,201]
    x203 = x[:,202]
    x204 = x[:,203]
    x205 = x[:,204]
    x206 = x[:,205]
    x207 = x[:,206]
    x208 = x[:,207]
    x209 = x[:,208]
    x210 = x[:,209]
    x211 = x[:,210]
    x212 = x[:,211]
    x213 = x[:,212]
    x214 = x[:,213]
    x215 = x[:,214]
    x216 = x[:,215]
    x217 = x[:,216]
    x218 = x[:,217]
    x219 = x[:,218]
    x220 = x[:,219]
    x221 = x[:,220]
    x222 = x[:,221]
    x223 = x[:,222]
    x224 = x[:,223]
    x225 = x[:,224]
    x226 = x[:,225]
    x227 = x[:,226]
    x228 = x[:,227]
    x229 = x[:,228]
    x230 = x[:,229]
    x231 = x[:,230]
    x232 = x[:,231]
    x233 = x[:,232]
    x234 = x[:,233]
    x235 = x[:,234]
    x236 = x[:,235]
    x237 = x[:,236]
    x238 = x[:,237]
    x239 = x[:,238]
    x240 = x[:,239]
    x241 = x[:,240]
    x242 = x[:,241]
    x243 = x[:,242]
    x244 = x[:,243]
    x245 = x[:,244]
    x246 = x[:,245]
    x247 = x[:,246]
    x248 = x[:,247]
    x249 = x[:,248]
    x250 = x[:,249]
    x251 = x[:,250]
    x252 = x[:,251]
    x253 = x[:,252]
    x254 = x[:,253]
    x255 = x[:,254]
    x256 = x[:,255]
    x257 = x[:,256]
    x258 = x[:,257]
    x259 = x[:,258]
    x260 = x[:,259]
    x261 = x[:,260]
    x262 = x[:,261]
    x263 = x[:,262]
    x264 = x[:,263]
    x265 = x[:,264]
    x266 = x[:,265]
    x267 = x[:,266]
    x268 = x[:,267]
    x269 = x[:,268]
    x270 = x[:,269]
    x271 = x[:,270]
    x272 = x[:,271]
    x273 = x[:,272]
    x274 = x[:,273]
    x275 = x[:,274]
    x276 = x[:,275]
    x277 = x[:,276]
    x278 = x[:,277]
    x279 = x[:,278]
    x280 = x[:,279]
    x281 = x[:,280]
    x282 = x[:,281]
    x283 = x[:,282]
    x284 = x[:,283]
    x285 = x[:,284]
    x286 = x[:,285]
    x287 = x[:,286]
    x288 = x[:,287]
    x289 = x[:,288]
    x290 = x[:,289]
    x291 = x[:,290]
    x292 = x[:,291]
    x293 = x[:,292]
    x294 = x[:,293]
    x295 = x[:,294]
    x296 = x[:,295]
    x297 = x[:,296]
    x298 = x[:,297]
    x299 = x[:,298]
    x300 = x[:,299]
    x301 = x[:,300]
    x302 = x[:,301]
    x303 = x[:,302]
    x304 = x[:,303]
    x305 = x[:,304]
    x306 = x[:,305]
    x307 = x[:,306]
    x308 = x[:,307]
    x309 = x[:,308]
    x310 = x[:,309]
    x311 = x[:,310]
    x312 = x[:,311]
    x313 = x[:,312]
    x314 = x[:,313]
    x315 = x[:,314]
    x316 = x[:,315]
    x317 = x[:,316]
    x318 = x[:,317]
    x319 = x[:,318]
    x320 = x[:,319]
    x321 = x[:,320]
    x322 = x[:,321]
    x323 = x[:,322]
    x324 = x[:,323]
    x325 = x[:,324]
    x326 = x[:,325]
    x327 = x[:,326]
    x328 = x[:,327]
    x329 = x[:,328]
    x330 = x[:,329]
    x331 = x[:,330]
    x332 = x[:,331]
    x333 = x[:,332]
    x334 = x[:,333]
    x335 = x[:,334]
    x336 = x[:,335]
    x337 = x[:,336]
    x338 = x[:,337]
    x339 = x[:,338]
    x340 = x[:,339]
    x341 = x[:,340]
    x342 = x[:,341]
    x343 = x[:,342]
    x344 = x[:,343]
    x345 = x[:,344]
    x346 = x[:,345]
    x347 = x[:,346]
    x348 = x[:,347]
    x349 = x[:,348]
    x350 = x[:,349]
    x351 = x[:,350]
    x352 = x[:,351]
    x353 = x[:,352]
    x354 = x[:,353]
    x355 = x[:,354]
    x356 = x[:,355]
    x357 = x[:,356]
    x358 = x[:,357]
    x359 = x[:,358]
    x360 = x[:,359]
    x361 = x[:,360]
    x362 = x[:,361]
    x363 = x[:,362]
    x364 = x[:,363]
    x365 = x[:,364]
    x366 = x[:,365]
    x367 = x[:,366]
    x368 = x[:,367]
    x369 = x[:,368]
    x370 = x[:,369]
    x371 = x[:,370]
    x372 = x[:,371]
    x373 = x[:,372]
    x374 = x[:,373]
    x375 = x[:,374]
    x376 = x[:,375]
    x377 = x[:,376]
    x378 = x[:,377]
    x379 = x[:,378]
    x380 = x[:,379]
    x381 = x[:,380]
    x382 = x[:,381]
    x383 = x[:,382]
    x384 = x[:,383]
    x385 = x[:,384]
    x386 = x[:,385]
    x387 = x[:,386]
    x388 = x[:,387]
    x389 = x[:,388]
    x390 = x[:,389]
    x391 = x[:,390]
    x392 = x[:,391]
    x393 = x[:,392]
    x394 = x[:,393]
    x395 = x[:,394]
    x396 = x[:,395]
    x397 = x[:,396]
    x398 = x[:,397]
    x399 = x[:,398]
    x400 = x[:,399]
    x401 = x[:,400]
    x402 = x[:,401]
    x403 = x[:,402]
    x404 = x[:,403]
    x405 = x[:,404]
    x406 = x[:,405]
    x407 = x[:,406]
    x408 = x[:,407]
    x409 = x[:,408]
    x410 = x[:,409]
    x411 = x[:,410]
    x412 = x[:,411]
    x413 = x[:,412]
    x414 = x[:,413]
    x415 = x[:,414]
    x416 = x[:,415]
    x417 = x[:,416]
    x418 = x[:,417]
    x419 = x[:,418]
    x420 = x[:,419]
    x421 = x[:,420]
    x422 = x[:,421]
    x423 = x[:,422]
    x424 = x[:,423]
    x425 = x[:,424]
    x426 = x[:,425]
    x427 = x[:,426]
    x428 = x[:,427]
    x429 = x[:,428]
    x430 = x[:,429]
    x431 = x[:,430]
    x432 = x[:,431]
    x433 = x[:,432]
    x434 = x[:,433]
    x435 = x[:,434]
    x436 = x[:,435]
    x437 = x[:,436]
    x438 = x[:,437]
    x439 = x[:,438]
    x440 = x[:,439]
    x441 = x[:,440]
    x442 = x[:,441]
    x443 = x[:,442]
    x444 = x[:,443]
    x445 = x[:,444]
    x446 = x[:,445]
    x447 = x[:,446]
    x448 = x[:,447]
    x449 = x[:,448]
    x450 = x[:,449]

    pos = np.array([x1 ,x2 ,x3 ,x4 ,x5 ,x6 ,x7 ,x8 ,x9 ,x10 ,x11 ,x12 ,x13 ,x14 ,x15 ,x16 ,x17 ,x18 ,x19 ,x20 ,x21 ,x22 ,x23 ,x24 ,x25 ,x26 ,x27 ,x28 ,x29 ,x30 ,x31 ,x32 ,x33 ,x34 ,x35 ,x36 ,x37 ,x38 ,x39 ,x40 ,x41 ,x42 ,x43 ,x44 ,x45 ,x46 ,x47 ,x48 ,x49 ,x50 ,x51 ,x52 ,x53 ,x54 ,x55 ,x56 ,x57 ,x58 ,x59 ,x60 ,x61 ,x62 ,x63 ,x64 ,x65 ,x66 ,x67 ,x68 ,x69 ,x70 ,x71 ,x72 ,x73 ,x74 ,x75 ,x76 ,x77 ,x78 ,x79 ,x80 ,x81 ,x82 ,x83 ,x84 ,x85 ,x86 ,x87 ,x88 ,x89 ,x90 ,x91 ,x92 ,x93 ,x94 ,x95 ,x96 ,x97 ,x98 ,x99 ,x100 ,x101 ,x102 ,x103 ,x104 ,x105 ,x106 ,x107 ,x108 ,x109 ,x110 ,x111 ,x112 ,x113 ,x114 ,x115 ,x116 ,x117 ,x118 ,x119 ,x120 ,x121 ,x122 ,x123 ,x124 ,x125 ,x126 ,x127 ,x128 ,x129 ,x130 ,x131 ,x132 ,x133 ,x134 ,x135 ,x136 ,x137 ,x138 ,x139 ,x140 ,x141 ,x142 ,x143 ,x144 ,x145 ,x146 ,x147 ,x148 ,x149 ,x150 ,x151 ,x152 ,x153 ,x154 ,x155 ,x156 ,x157 ,x158 ,x159 ,x160 ,x161 ,x162 ,x163 ,x164 ,x165 ,x166 ,x167 ,x168 ,x169 ,x170 ,x171 ,x172 ,x173 ,x174 ,x175 ,x176 ,x177 ,x178 ,x179 ,x180 ,x181 ,x182 ,x183 ,x184 ,x185 ,x186 ,x187 ,x188 ,x189 ,x190 ,x191 ,x192 ,x193 ,x194 ,x195 ,x196 ,x197 ,x198 ,x199 ,x200 ,x201 ,x202 ,x203 ,x204 ,x205 ,x206 ,x207 ,x208 ,x209 ,x210 ,x211 ,x212 ,x213 ,x214 ,x215 ,x216 ,x217 ,x218 ,x219 ,x220 ,x221 ,x222 ,x223 ,x224 ,x225 ,x226 ,x227 ,x228 ,x229 ,x230 ,x231 ,x232 ,x233 ,x234 ,x235 ,x236 ,x237 ,x238 ,x239 ,x240 ,x241 ,x242 ,x243 ,x244 ,x245 ,x246 ,x247 ,x248 ,x249 ,x250 ,x251 ,x252 ,x253 ,x254 ,x255 ,x256 ,x257 ,x258 ,x259 ,x260 ,x261 ,x262 ,x263 ,x264 ,x265 ,x266 ,x267 ,x268 ,x269 ,x270 ,x271 ,x272 ,x273 ,x274 ,x275 ,x276 ,x277 ,x278 ,x279 ,x280 ,x281 ,x282 ,x283 ,x284 ,x285 ,x286 ,x287 ,x288 ,x289 ,x290 ,x291 ,x292 ,x293 ,x294 ,x295 ,x296 ,x297 ,x298 ,x299 ,x300 ,x301 ,x302 ,x303 ,x304 ,x305 ,x306 ,x307 ,x308 ,x309 ,x310 ,x311 ,x312 ,x313 ,x314 ,x315 ,x316 ,x317 ,x318 ,x319 ,x320 ,x321 ,x322 ,x323 ,x324 ,x325 ,x326 ,x327 ,x328 ,x329 ,x330 ,x331 ,x332 ,x333 ,x334 ,x335 ,x336 ,x337 ,x338 ,x339 ,x340 ,x341 ,x342 ,x343 ,x344 ,x345 ,x346 ,x347 ,x348 ,x349 ,x350 ,x351 ,x352 ,x353 ,x354 ,x355 ,x356 ,x357 ,x358 ,x359 ,x360 ,x361 ,x362 ,x363 ,x364 ,x365 ,x366 ,x367 ,x368 ,x369 ,x370 ,x371 ,x372 ,x373 ,x374 ,x375 ,x376 ,x377 ,x378 ,x379 ,x380 ,x381 ,x382 ,x383 ,x384 ,x385 ,x386 ,x387 ,x388 ,x389 ,x390 ,x391 ,x392 ,x393 ,x394 ,x395 ,x396 ,x397 ,x398 ,x399 ,x400 ,x401 ,x402 ,x403 ,x404 ,x405 ,x406 ,x407 ,x408 ,x409 ,x410 ,x411 ,x412 ,x413 ,x414 ,x415 ,x416 ,x417 ,x418 ,x419 ,x420 ,x421 ,x422 ,x423 ,x424 ,x425 ,x426 ,x427 ,x428 ,x429 ,x430 ,x431 ,x432 ,x433 ,x434 ,x435 ,x436 ,x437 ,x438 ,x439 ,x440 ,x441 ,x442 ,x443 ,x444 ,x445 ,x446 ,x447 ,x448 ,x449, x450])
    return su.svd_norm2(pos, s1=s1)

def calc():    
    cell_len = atoms.get_cell()[0,0]
    x0 = atoms.get_positions()
    bounds = (0., cell_len)
    X_init = x0.reshape(-1)

    #bounds
    kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)
    bds = [{'name': 'keks1', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks2', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks3', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks4', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks5', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks6', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks7', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks8', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks9', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks10', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks11', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks12', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks13', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks14', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks15', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks16', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks17', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks18', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks19', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks20', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks21', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks22', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks23', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks24', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks25', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks26', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks27', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks28', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks29', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks30', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks31', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks32', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks33', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks34', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks35', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks36', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks37', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks38', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks39', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks40', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks41', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks42', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks43', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks44', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks45', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks46', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks47', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks48', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks49', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks50', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks51', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks52', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks53', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks54', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks55', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks56', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks57', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks58', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks59', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks60', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks61', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks62', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks63', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks64', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks65', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks66', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks67', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks68', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks69', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks70', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks71', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks72', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks73', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks74', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks75', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks76', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks77', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks78', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks79', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks80', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks81', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks82', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks83', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks84', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks85', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks86', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks87', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks88', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks89', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks90', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks91', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks92', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks93', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks94', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks95', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks96', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks97', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks98', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks99', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks100', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks101', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks102', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks103', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks104', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks105', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks106', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks107', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks108', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks109', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks110', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks111', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks112', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks113', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks114', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks115', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks116', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks117', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks118', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks119', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks120', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks121', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks122', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks123', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks124', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks125', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks126', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks127', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks128', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks129', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks130', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks131', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks132', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks133', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks134', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks135', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks136', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks137', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks138', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks139', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks140', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks141', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks142', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks143', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks144', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks145', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks146', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks147', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks148', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks149', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks150', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks151', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks152', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks153', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks154', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks155', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks156', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks157', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks158', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks159', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks160', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks161', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks162', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks163', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks164', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks165', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks166', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks167', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks168', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks169', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks170', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks171', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks172', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks173', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks174', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks175', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks176', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks177', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks178', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks179', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks180', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks181', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks182', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks183', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks184', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks185', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks186', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks187', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks188', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks189', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks190', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks191', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks192', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks193', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks194', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks195', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks196', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks197', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks198', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks199', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks200', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks201', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks202', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks203', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks204', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks205', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks206', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks207', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks208', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks209', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks210', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks211', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks212', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks213', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks214', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks215', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks216', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks217', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks218', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks219', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks220', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks221', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks222', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks223', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks224', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks225', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks226', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks227', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks228', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks229', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks230', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks231', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks232', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks233', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks234', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks235', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks236', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks237', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks238', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks239', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks240', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks241', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks242', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks243', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks244', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks245', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks246', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks247', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks248', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks249', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks250', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks251', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks252', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks253', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks254', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks255', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks256', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks257', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks258', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks259', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks260', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks261', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks262', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks263', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks264', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks265', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks266', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks267', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks268', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks269', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks270', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks271', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks272', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks273', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks274', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks275', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks276', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks277', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks278', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks279', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks280', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks281', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks282', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks283', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks284', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks285', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks286', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks287', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks288', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks289', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks290', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks291', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks292', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks293', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks294', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks295', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks296', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks297', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks298', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks299', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks300', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks301', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks302', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks303', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks304', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks305', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks306', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks307', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks308', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks309', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks310', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks311', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks312', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks313', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks314', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks315', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks316', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks317', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks318', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks319', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks320', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks321', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks322', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks323', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks324', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks325', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks326', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks327', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks328', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks329', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks330', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks331', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks332', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks333', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks334', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks335', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks336', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks337', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks338', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks339', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks340', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks341', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks342', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks343', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks344', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks345', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks346', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks347', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks348', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks349', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks350', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks351', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks352', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks353', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks354', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks355', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks356', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks357', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks358', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks359', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks360', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks361', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks362', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks363', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks364', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks365', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks366', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks367', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks368', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks369', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks370', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks371', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks372', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks373', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks374', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks375', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks376', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks377', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks378', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks379', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks380', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks381', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks382', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks383', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks384', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks385', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks386', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks387', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks388', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks389', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks390', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks391', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks392', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks393', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks394', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks395', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks396', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks397', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks398', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks399', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks400', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks401', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks402', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks403', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks404', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks405', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks406', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks407', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks408', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks409', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks410', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks411', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks412', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks413', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks414', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks415', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks416', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks417', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks418', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks419', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks420', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks421', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks422', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks423', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks424', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks425', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks426', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks427', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks428', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks429', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks430', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks431', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks432', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks433', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks434', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks435', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks436', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks437', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks438', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks439', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks440', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks441', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks442', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks443', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks444', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks445', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks446', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks447', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks448', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks449', 'type': 'continuous', 'domain': bounds},
    {'name': 'keks450', 'type': 'continuous', 'domain': bounds}]

    t0 = time.time()
    
    # define optimization parameters:
    optimizer = BayesianOptimization(black_box_function, domain=bds,
                                     model_type='GP',
                                     kernel=kernel,
                                     acquisition_type ='EI',
                                     acquisition_jitter = 0.01)

    # run the optimization and plots:
    optimizer.run_optimization(max_iter=5000)
    optimizer.plot_acquisition()
    optimizer.plot_convergence()

    xopt = optimizer.x_opt
    
    atoms_res = atoms.copy()
    atoms_res.set_positions(np.reshape(xopt,(-1,3)))
    
    # the resulting struct is saved in a folder
    filename = "gpyoptpbc2_" + n + ".cfg"
    ase.io.write("res_structs/" + filename, atoms_res)
    return 0 # return value can be ignored

calc()