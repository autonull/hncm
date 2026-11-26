# LM (P0a): Tiny Shakespeare

- corpus `shakespeare` sha `86c4e6aa9db7`, train 1,003,836 B, test 111,558 B
- model 1,143,680 params (d=128, 4 layers), 2000 steps, ngram-joint, seq 256
- pack 256 units / 1,409 B installed from the train split

| metric | value |
|---|---:|
| train loss (nats/elem, last 50) | 2.944 |
| train bpb (teacher-forced, 128 kB) | 2.412 |
| test payload bpb | 2.775 |
| test payload bytes | 38,694 |
| test coverage | 0.690 |
| proof | 4,000 units / 14,173 B, OK |
| encode | 6.3 s (17.3 kB/s) |

`shared` is not reported here.  The pack charge is unscaled at this
size: 1,409 B / 111,558 B = 0.101 bpb.  The text bpb claim needs the 1 MB enwik8 test, not this slice.

## Sample after `ROMEO:`

```
ROMEO:F MARCEANPETERUS METARK:
I? isek'd yould be a faise the hind adier coot
will whick is makent, but prochse not!
note's day my masten dure amoning fellor,
Wher's your moard death to provice there and Engy
Th him, deep of plose shall vattery for airce,
Whe ay news that we sus Spielc, as jisalage
Was you, himch buts is rantunc mistlepst hurge sortm
And uso for I jointiou
ShaIn he wee's hap, news, is a
```

## Provenance: every emitted span

copies 105, neural 132

| start | end | tag | bytes |
|---:|---:|---|---|
| 0 | 1 | `~neural` | `b'F'` |
| 1 | 2 | `~neural` | `b' '` |
| 2 | 3 | `~neural` | `b'M'` |
| 3 | 4 | `~neural` | `b'A'` |
| 4 | 5 | `~neural` | `b'R'` |
| 5 | 6 | `~neural` | `b'C'` |
| 6 | 7 | `~neural` | `b'E'` |
| 7 | 8 | `~neural` | `b'A'` |
| 8 | 9 | `~neural` | `b'N'` |
| 9 | 10 | `~neural` | `b'P'` |
| 10 | 11 | `~neural` | `b'E'` |
| 11 | 12 | `~neural` | `b'T'` |
| 12 | 13 | `~neural` | `b'E'` |
| 13 | 14 | `~neural` | `b'R'` |
| 14 | 15 | `~neural` | `b'U'` |
| 15 | 16 | `~neural` | `b'S'` |
| 16 | 17 | `~neural` | `b' '` |
| 17 | 18 | `~neural` | `b'M'` |
| 18 | 19 | `~neural` | `b'E'` |
| 19 | 20 | `~neural` | `b'T'` |
| 20 | 21 | `~neural` | `b'A'` |
| 21 | 22 | `~neural` | `b'R'` |
| 22 | 23 | `~neural` | `b'K'` |
| 23 | 25 | `copy@240` | `b':\n'` |
| 25 | 26 | `~neural` | `b'I'` |
| 26 | 27 | `~neural` | `b'?'` |
| 27 | 28 | `~neural` | `b' '` |
| 28 | 30 | `copy@325` | `b'is'` |
| 30 | 31 | `~neural` | `b'e'` |
| 31 | 32 | `~neural` | `b'k'` |
| 32 | 33 | `~neural` | `b"'"` |
| 33 | 35 | `copy@72` | `b'd '` |
| 35 | 38 | `copy@204` | `b'you'` |
| 38 | 39 | `~neural` | `b'l'` |
| 39 | 41 | `copy@72` | `b'd '` |
| 41 | 43 | `copy@1130` | `b'be'` |
| 43 | 45 | `copy@93` | `b' a'` |
| 45 | 47 | `copy@444` | `b' f'` |
| 47 | 48 | `~neural` | `b'a'` |
| 48 | 49 | `~neural` | `b'i'` |
| 49 | 51 | `copy@621` | `b'se'` |
| 51 | 53 | `copy@29` | `b' t'` |
| 53 | 56 | `copy@63` | `b'he '` |
| 56 | 58 | `copy@387` | `b'hi'` |
| 58 | 61 | `copy@97` | `b'nd '` |
| 61 | 62 | `~neural` | `b'a'` |
| 62 | 63 | `~neural` | `b'd'` |
| 63 | 64 | `~neural` | `b'i'` |
| 64 | 67 | `copy@452` | `b'er '` |
| 67 | 68 | `~neural` | `b'c'` |
| 68 | 69 | `~neural` | `b'o'` |
| 69 | 71 | `copy@1082` | `b'ot'` |
| 71 | 72 | `~neural` | `b'\n'` |
| 72 | 73 | `~neural` | `b'w'` |
| 73 | 77 | `copy@1245` | `b'ill '` |
| 77 | 78 | `~neural` | `b'w'` |
| 78 | 80 | `copy@387` | `b'hi'` |
| 80 | 81 | `~neural` | `b'c'` |
| 81 | 82 | `~neural` | `b'k'` |
| 82 | 86 | `copy@821` | `b' is '` |
| 86 | 87 | `~neural` | `b'm'` |
| 87 | 88 | `~neural` | `b'a'` |
| 88 | 89 | `~neural` | `b'k'` |
| 89 | 91 | `copy@313` | `b'en'` |
| 91 | 92 | `~neural` | `b't'` |
| 92 | 94 | `copy@83` | `b', '` |
| 94 | 95 | `~neural` | `b'b'` |
| 95 | 98 | `copy@1290` | `b'ut '` |
| 98 | 99 | `~neural` | `b'p'` |
| 99 | 101 | `copy@1330` | `b'ro'` |
| 101 | 102 | `~neural` | `b'c'` |
| 102 | 103 | `~neural` | `b'h'` |
| 103 | 105 | `copy@621` | `b'se'` |
| 105 | 109 | `copy@543` | `b' not'` |
| 109 | 110 | `~neural` | `b'!'` |
| 110 | 111 | `~neural` | `b'\n'` |
| 111 | 114 | `copy@914` | `b'not'` |
| 114 | 115 | `~neural` | `b'e'` |
| 115 | 116 | `~neural` | `b"'"` |
| 116 | 118 | `copy@68` | `b's '` |
| 118 | 119 | `~neural` | `b'd'` |
| 119 | 120 | `~neural` | `b'a'` |
| 120 | 122 | `copy@152` | `b'y '` |
| 122 | 125 | `copy@724` | `b'my '` |
| 125 | 126 | `~neural` | `b'm'` |
| 126 | 128 | `copy@1281` | `b'as'` |
| 128 | 130 | `copy@982` | `b'te'` |
| 130 | 132 | `copy@224` | `b'n '` |
| 132 | 133 | `~neural` | `b'd'` |
| 133 | 135 | `copy@683` | `b'ur'` |
| 135 | 138 | `copy@1043` | `b'e a'` |
| 138 | 139 | `~neural` | `b'm'` |
| 139 | 141 | `copy@448` | `b'on'` |
| 141 | 145 | `copy@438` | `b'ing '` |
| 145 | 146 | `~neural` | `b'f'` |
| 146 | 147 | `~neural` | `b'e'` |
| 147 | 149 | `copy@481` | `b'll'` |
| 149 | 151 | `copy@236` | `b'or'` |
| 151 | 153 | `copy@653` | `b',\n'` |
| 153 | 154 | `~neural` | `b'W'` |
| 154 | 157 | `copy@277` | `b'her'` |
| 157 | 158 | `~neural` | `b"'"` |
| 158 | 160 | `copy@68` | `b's '` |
| 160 | 165 | `copy@569` | `b'your '` |
| 165 | 166 | `~neural` | `b'm'` |
| 166 | 167 | `~neural` | `b'o'` |
| 167 | 169 | `copy@374` | `b'ar'` |
| 169 | 171 | `copy@72` | `b'd '` |
| 171 | 172 | `~neural` | `b'd'` |
| 172 | 174 | `copy@477` | `b'ea'` |
| 174 | 177 | `copy@485` | `b'th '` |
| 177 | 180 | `copy@253` | `b'to '` |
| 180 | 181 | `~neural` | `b'p'` |
| 181 | 183 | `copy@1330` | `b'ro'` |
| 183 | 184 | `~neural` | `b'v'` |
| 184 | 185 | `~neural` | `b'i'` |
| 185 | 186 | `~neural` | `b'c'` |
| 186 | 189 | `copy@362` | `b'e t'` |
| 189 | 193 | `copy@908` | `b'here'` |
| 193 | 198 | `copy@76` | `b' and '` |
| 198 | 199 | `~neural` | `b'E'` |
| 199 | 201 | `copy@774` | `b'ng'` |
| 201 | 202 | `~neural` | `b'y'` |
| 202 | 205 | `copy@599` | `b'\nTh'` |
| 205 | 209 | `copy@1101` | `b' him'` |
| 209 | 211 | `copy@83` | `b', '` |
| 211 | 212 | `~neural` | `b'd'` |
| 212 | 214 | `copy@1277` | `b'ee'` |
| 214 | 215 | `~neural` | `b'p'` |
| 215 | 219 | `copy@176` | `b' of '` |
| 219 | 220 | `~neural` | `b'p'` |
| 220 | 221 | `~neural` | `b'l'` |
| 221 | 222 | `~neural` | `b'o'` |
| 222 | 224 | `copy@621` | `b'se'` |
| 224 | 231 | `copy@1073` | `b' shall '` |
| 231 | 232 | `~neural` | `b'v'` |
| 232 | 234 | `copy@258` | `b'at'` |
| 234 | 236 | `copy@982` | `b'te'` |
| 236 | 237 | `~neural` | `b'r'` |
| 237 | 239 | `copy@152` | `b'y '` |
| 239 | 243 | `copy@1052` | `b'for '` |
| 243 | 244 | `~neural` | `b'a'` |
| 244 | 245 | `~neural` | `b'i'` |
| 245 | 246 | `~neural` | `b'r'` |
| 246 | 247 | `~neural` | `b'c'` |
| 247 | 249 | `copy@1273` | `b'e,'` |
| 249 | 250 | `~neural` | `b'\n'` |
| 250 | 251 | `~neural` | `b'W'` |
| 251 | 254 | `copy@63` | `b'he '` |
| 254 | 255 | `~neural` | `b'a'` |
| 255 | 257 | `copy@152` | `b'y '` |
| 257 | 259 | `copy@1159` | `b'ne'` |
| 259 | 260 | `~neural` | `b'w'` |
| 260 | 264 | `copy@1262` | `b's th'` |
| 264 | 267 | `copy@268` | `b'at '` |
| 267 | 268 | `~neural` | `b'w'` |
| 268 | 271 | `copy@1011` | `b'e s'` |
| 271 | 272 | `~neural` | `b'u'` |
| 272 | 274 | `copy@68` | `b's '` |
| 274 | 275 | `~neural` | `b'S'` |
| 275 | 276 | `~neural` | `b'p'` |
| 276 | 277 | `~neural` | `b'i'` |
| 277 | 278 | `~neural` | `b'e'` |
| 278 | 279 | `~neural` | `b'l'` |
| 279 | 280 | `~neural` | `b'c'` |
| 280 | 283 | `copy@1285` | `b', a'` |
| 283 | 285 | `copy@68` | `b's '` |
| 285 | 286 | `~neural` | `b'j'` |
| 286 | 288 | `copy@325` | `b'is'` |
| 288 | 290 | `copy@1039` | `b'al'` |
| 290 | 291 | `~neural` | `b'a'` |
| 291 | 292 | `~neural` | `b'g'` |
| 292 | 293 | `~neural` | `b'e'` |
| 293 | 294 | `~neural` | `b'\n'` |
| 294 | 295 | `~neural` | `b'W'` |
| 295 | 297 | `copy@1281` | `b'as'` |
| 297 | 301 | `copy@87` | `b' you'` |
| 301 | 303 | `copy@83` | `b', '` |
| 303 | 305 | `copy@387` | `b'hi'` |
| 305 | 306 | `~neural` | `b'm'` |
| 306 | 307 | `~neural` | `b'c'` |
| 307 | 309 | `copy@791` | `b'h '` |
| 309 | 310 | `~neural` | `b'b'` |
| 310 | 311 | `~neural` | `b'u'` |
| 311 | 312 | `~neural` | `b't'` |
| 312 | 314 | `copy@68` | `b's '` |
| 314 | 317 | `copy@166` | `b'is '` |
| 317 | 318 | `~neural` | `b'r'` |
| 318 | 320 | `copy@156` | `b'an'` |
| 320 | 321 | `~neural` | `b't'` |
| 321 | 322 | `~neural` | `b'u'` |
| 322 | 323 | `~neural` | `b'n'` |
| 323 | 324 | `~neural` | `b'c'` |
| 324 | 326 | `copy@136` | `b' m'` |
| 326 | 328 | `copy@325` | `b'is'` |
| 328 | 329 | `~neural` | `b't'` |
| 329 | 331 | `copy@972` | `b'le'` |
| 331 | 332 | `~neural` | `b'p'` |
| 332 | 335 | `copy@702` | `b'st '` |
| 335 | 336 | `~neural` | `b'h'` |
| 336 | 338 | `copy@683` | `b'ur'` |
| 338 | 339 | `~neural` | `b'g'` |
| 339 | 342 | `copy@1011` | `b'e s'` |
| 342 | 344 | `copy@236` | `b'or'` |
| 344 | 345 | `~neural` | `b't'` |
| 345 | 346 | `~neural` | `b'm'` |
| 346 | 351 | `copy@391` | `b'\nAnd '` |
| 351 | 352 | `~neural` | `b'u'` |
| 352 | 353 | `~neural` | `b's'` |
| 353 | 355 | `copy@273` | `b'o '` |
| 355 | 359 | `copy@1052` | `b'for '` |
| 359 | 361 | `copy@1007` | `b'I '` |
| 361 | 362 | `~neural` | `b'j'` |
| 362 | 363 | `~neural` | `b'o'` |
| 363 | 365 | `copy@140` | `b'in'` |
| 365 | 366 | `~neural` | `b't'` |
| 366 | 367 | `~neural` | `b'i'` |
| 367 | 369 | `copy@102` | `b'ou'` |
| 369 | 370 | `~neural` | `b'\n'` |
| 370 | 371 | `~neural` | `b'S'` |
| 371 | 373 | `copy@191` | `b'ha'` |
| 373 | 374 | `~neural` | `b'I'` |
| 374 | 376 | `copy@224` | `b'n '` |
| 376 | 379 | `copy@63` | `b'he '` |
| 379 | 380 | `~neural` | `b'w'` |
| 380 | 382 | `copy@1277` | `b'ee'` |
| 382 | 383 | `~neural` | `b"'"` |
| 383 | 385 | `copy@68` | `b's '` |
| 385 | 387 | `copy@191` | `b'ha'` |
| 387 | 388 | `~neural` | `b'p'` |
| 388 | 390 | `copy@83` | `b', '` |
| 390 | 392 | `copy@1159` | `b'ne'` |
| 392 | 393 | `~neural` | `b'w'` |
| 393 | 394 | `~neural` | `b's'` |
| 394 | 396 | `copy@83` | `b', '` |
| 396 | 399 | `copy@166` | `b'is '` |
| 399 | 400 | `~neural` | `b'a'` |
