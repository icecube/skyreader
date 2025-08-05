# CHANGELOG



## v1.4.6 (2025-08-05)

###  

* Serialize/Deserialize `nan` for JSON (#52)

`nan` is a valid Python value, but it is not a valid JSON value. Build
out serialization for this.

---------

Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`39e7977`](https://github.com/icecube/skyreader/commit/39e79777299d5d1c71c355985dde3a39830e9b88))


## v1.4.5 (2025-05-15)

###  

* Add information to the header of the reco fits file (rectangular errors at 50% + contour areas at 50% and 90%) (#50)

Hi,

I propose a slight modification to the function that generates zoomed
plots from reconstructions.
The modification would allow it to save RA and Dec, as well as the
corresponding 50% and 90% rectangular and circular areas.

https://github.com/icecube/skyreader/blob/40a75068b486225fa02c433caf6cdd76720ddf1e/skyreader/plot/plot.py#L729

This change would make it easier to manage reconstruction outputs in
general, and is necessary for posting reconstruction results on Slack
channels as part of the automation I am currently working on.

---------

Co-authored-by: github-actions &lt;github-actions@github.com&gt;
Co-authored-by: Angela &lt;angela@mobile-222.shadow2.airub.net&gt;
Co-authored-by: ric-evans &lt;emejqz@gmail.com&gt; ([`6aec865`](https://github.com/icecube/skyreader/commit/6aec865588c80171190ed7c2911c7867bf3cc8f6))


## v1.4.4 (2025-04-15)

###  

* Use `pypa/gh-action-pypi-publish@v1.12.4` ([`82f14e6`](https://github.com/icecube/skyreader/commit/82f14e6af4b318c1ea38911d97fc5e9e13f35107))


## v1.4.3 (2025-03-27)

###  

* Update Fermi catalog: new branch analogous to the other but that don&#39;t produce conflicts (shift from using setup.cfg to pyproject.toml) (#51)

Co-authored-by: github-actions &lt;github-actions@github.com&gt;
Co-authored-by: Angela &lt;angela@mobile-222.shadow2.airub.net&gt; ([`729b2ff`](https://github.com/icecube/skyreader/commit/729b2ffd9beda13b4281599a8015b96d7ddd5ae4))


## v1.4.2 (2025-03-14)

###  

* Patch has_minimal_metadata and fix deprecation (#49) ([`49f26f5`](https://github.com/icecube/skyreader/commit/49f26f5075750f95f7128e84f540483920b47c41))


## v1.4.1 (2025-03-14)

###  

* Default to version 1 (#48) ([`ed2bf9c`](https://github.com/icecube/skyreader/commit/ed2bf9c1d9da95cc83578975c07c4dc9393c04d4))


## v1.4.0 (2025-03-13)

###  

* commit to update wipac-dev-tools ([`7cd5efc`](https://github.com/icecube/skyreader/commit/7cd5efcf616c525364805a3afccc7d7a056ab4cd))

### [minor]

* [minor] Write reconstructed position and time (#47)

Add x, y and z position from the reco `I3Position` and the corresponding
time to the result file for each scanned pixel. This allows to display
the reconstructed track later for visualization. I will make a
corresponding PR in `skymap_scanner` as well.

Adds a data format version flag to allow backwards compatibility with reading existing results using the previous version.

---------

Co-authored-by: github-actions &lt;github-actions@github.com&gt;
Co-authored-by: Tianlu Yuan &lt;5412915+tianluyuan@users.noreply.github.com&gt; ([`06918f3`](https://github.com/icecube/skyreader/commit/06918f3932f6a91d54a81334eccaef674bbf8827))


## v1.3.6 (2025-02-13)

###  

* Updates to CI and Packaging - 12 ([`dc05fbe`](https://github.com/icecube/skyreader/commit/dc05fbe80cff23620befe548cd6d1eb58a78cc63))

* Updates to CI and Packaging - 11 ([`adb2a91`](https://github.com/icecube/skyreader/commit/adb2a91d65fc82389d882dff6e22c178538d8c34))

* Updates to CI and Packaging - 10 ([`cf15978`](https://github.com/icecube/skyreader/commit/cf15978085671743b14bc7bf8a4ec4603b9b0712))

* Merge remote-tracking branch &#39;origin/main&#39; ([`2439ca8`](https://github.com/icecube/skyreader/commit/2439ca8d0af300a6a73efffddb2c2cda1df1184a))

* Updates to CI and Packaging - 9 ([`d305173`](https://github.com/icecube/skyreader/commit/d3051739273496fa5bc2529a143c682b5ef5ccb3))


## v1.3.5 (2025-02-13)

###  

* Updates to CI and Packaging - 8 ([`0791bc2`](https://github.com/icecube/skyreader/commit/0791bc2a9481550d2d03539f8458953ebdcd9ea0))

* Updates to CI and Packaging - 7 ([`d04eb9c`](https://github.com/icecube/skyreader/commit/d04eb9c642689902156e3a35ddcb1a20abdf8008))


## v1.3.4 (2025-02-13)

###  

* Updates to CI and Packaging - 6 (#46) ([`2b4ca1b`](https://github.com/icecube/skyreader/commit/2b4ca1b3443ee17a9a0d0db323e5efcaffaa3dc2))

* Updates to CI and Packaging - 5 ([`9ca4e17`](https://github.com/icecube/skyreader/commit/9ca4e170666748c26046477684da6afd96eecbac))

* Updates to CI and Packaging - 4 ([`ccc7368`](https://github.com/icecube/skyreader/commit/ccc736895d460b411c17ae609e4965818901871c))

* Updates to CI and Packaging - 3 (#45) ([`a7e4701`](https://github.com/icecube/skyreader/commit/a7e470118de4b624da4999200cfbfc53f5001610))

* Updates to CI and Packaging - 2 (#44) ([`3208053`](https://github.com/icecube/skyreader/commit/3208053efe3a334e27a0355680bf8c2ae5f9d5e3))

* Updates to CI and Packaging (#43)

Co-authored-by: github-actions &lt;github-actions@github.com&gt;
Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`b69f3aa`](https://github.com/icecube/skyreader/commit/b69f3aa77fb4fa2dbe4d98c1603be8b248d23424))


## v1.3.3 (2025-02-07)

###  

* &lt;bot&gt; update dependencies*.log files(s) ([`3fccfa1`](https://github.com/icecube/skyreader/commit/3fccfa10c018839ec5e81e9c3bb110e4a2ae5610))

* Compatibility with lvk (#41)

To be more compatible with LVK, I:
- Changed the unit in the multiorder maps from deg-2 to sr-1;
- The unit is stored now in `TUNIT2` in the header;
- Changed the column name from `PROBABILITY DENSITY [deg-2]` to
`[PROBDENSITY]`.

Moreover, in `plot.py` there were problems bacause the new matplotlib
doesn&#39;t support anymore the attribute `QuadContourSet.collections`. This
affects only `create_plot` when `dozoom` is set to `True`. I tried by
using `get_paths()` to fix the problem, but I am not totally sure it is
really doing what it should. At least in general it is not anymore
giving error. However, is the option `dozoom=True` still used? It seems
to me only redundant because there is already `create_plot_zoomed`.
Could we get rid of the option `dozoom=True`?

---------

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`96c1751`](https://github.com/icecube/skyreader/commit/96c1751b9f8f70fdc2a2496c05cdc26e8f525e41))


## v1.3.2 (2024-12-12)

###  

* Save contour files to specified output directory (#40) ([`55e9b2f`](https://github.com/icecube/skyreader/commit/55e9b2f25c6a6d82b455ecc9475fdd15c7ca836a))


## v1.3.1 (2024-12-11)

###  

* Multiorder maps (#39)

With this pull request I would like to include in skyreader an
implementation to save scans as multiorder maps.
The idea is to save exactly the same pixels as the ones which were
scanned, and for each report the density of probability (or density of
llh [deg-2] if a llh map is desired).

To do this, I needed to add some functions in `handle_map_data.py`, as
for some directions there are multiple pixels scanned with different
nsides. In these cases, the pixels with the bigger nside need to be
ignored.

Moreover, I added some logic in `extract_map` to fill the bigger nside
with empty pixels. This is necessary to produce multiorder maps for
pointed scans as well. It does not cause problems in the rest of the
processing as further on the empty pixels are filled with the smallest
registered probability.

Happy to receive feedback about this!

---------

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`21ec1d6`](https://github.com/icecube/skyreader/commit/21ec1d6a6247ff944268cc43b8ad6593cb00eabc))


## v1.3.0 (2024-10-14)

###  

* &lt;bot&gt; update dependencies*.log files(s) ([`1d75b32`](https://github.com/icecube/skyreader/commit/1d75b3201679485c29102a10fa5f835b4e003142))

### [minor]

* [minor] Probability map convolution (#37)

Implemented switch to probability maps and convolution with 0.2 deg
gaussian for SplineMPE recos.
To produce an old llh map there is the option to specify `llh_map=True`.
The convolution is optional and is implemented with a built-in function
of healpy. In the comments there is already the logic for multiorder
maps.

---------

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt;
Co-authored-by: Ric Evans &lt;19216225+ric-evans@users.noreply.github.com&gt; ([`2ad316b`](https://github.com/icecube/skyreader/commit/2ad316b8b552722fd8372dad3acbb0c319ef40ef))


## v1.2.12 (2024-10-04)

###  

* Bump min Python to 3.9 (#38) ([`93bd67f`](https://github.com/icecube/skyreader/commit/93bd67fab0bacfe3ecfae4020379bd4d4a52ccd4))


## v1.2.11 (2024-08-02)

###  

* &lt;bot&gt; update dependencies*.log files(s) ([`439e8f9`](https://github.com/icecube/skyreader/commit/439e8f970cfe8f2ffd17a5c58cbe1d4d4f812fcb))

* Area calculation spherical space (#36)

* &lt;bot&gt; update dependencies*.log files(s)

* correct area calculation for spherical space and drop pixel count

* removed # flake8: noqa

* added again # flake8: noqa

---------

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`43ae13f`](https://github.com/icecube/skyreader/commit/43ae13f20fdbb7c06060c4ab87a9cc76c65c7d88))


## v1.2.10 (2024-07-21)

###  

* &lt;bot&gt; update dependencies*.log files(s) ([`eacbc29`](https://github.com/icecube/skyreader/commit/eacbc294a002b5d7ce2294cc68982b5484bd1d84))

* All plotting outputs to the same directory (#32) ([`8cee17e`](https://github.com/icecube/skyreader/commit/8cee17ebc7da42b69c60da5e3908fa13fb17872d))


## v1.2.9 (2024-04-25)

###  

* Update function for plotting map with the last available Fermi LAT source catalog (#33) ([`8fbce83`](https://github.com/icecube/skyreader/commit/8fbce8301773decaad3bc1278226ad10c21cfc3a))


## v1.2.8 (2023-11-14)

###  

* Zoomed plot title (#31)

Added `bbox_inches=&#39;tight&#39; in savefig`, so now in the saved pdf appears the title as well. ([`27401e5`](https://github.com/icecube/skyreader/commit/27401e50acfa354551cef9c02903e713d63ae3b2))


## v1.2.7 (2023-10-22)

###  

* Split plotting functionality from `SkyScanResult` (#28)

* remove SkyScanResult attributes and methods

* &lt;bot&gt; update requirements-examples.txt

* &lt;bot&gt; update requirements-tests.txt

* &lt;bot&gt; update requirements.txt

* remove plotting stuff

* mypy partial compliance

* mypy compliance

* test compliance

* documentation

* add __all__ for flake8 compliance

* move logger

* logging no longer needed

* bump ci

* add `mypy` extra for ci

* add optional output dir

* &lt;bot&gt; update setup.cfg

* &lt;bot&gt; update .gitignore

* &lt;bot&gt; update dependencies*.log files(s)

---------

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt;
Co-authored-by: Ric Evans &lt;ric@evans-jacquez.com&gt; ([`dcb25a0`](https://github.com/icecube/skyreader/commit/dcb25a0408c4986d1a4418bb1dba3f4cb9d84e75))

* [split: restore `result.py`] ([`0de5408`](https://github.com/icecube/skyreader/commit/0de5408cb4c2921a289caee7bf74046ec7c82f16))

* [split: add `plot/plot.py`] ([`10c9571`](https://github.com/icecube/skyreader/commit/10c957151699e13312cf9660568cd0b90701e4d8))

* [split: temp] ([`f10a62e`](https://github.com/icecube/skyreader/commit/f10a62e2a82d74a422af0637dd3dd146f4d5566c))

* [split: make `plot/plot.py`] ([`effd960`](https://github.com/icecube/skyreader/commit/effd960dd801eb1593fa05c2fe85e0e13d21b2f6))


## v1.2.6 (2023-10-13)

###  

* New option for circular contours with radii sent as parameters (#25)

* deprecate log_func

* &lt;bot&gt; update requirements-examples.txt

* &lt;bot&gt; update requirements-tests.txt

* &lt;bot&gt; update requirements.txt

* deprecate upload-func

* deprecate further slack posting logic

* further remove deprecated logic

* assume we always save plots

* single source of truth for inches/dpi

* remove unused vars

* remove import io

* update example

* de-type plotting function

* some notes from mypy

* mypy readiness step

* &lt;bot&gt; update setup.cfg

* &lt;bot&gt; update requirements-examples.txt

* &lt;bot&gt; update requirements-tests.txt

* &lt;bot&gt; update requirements.txt

* Testing conection branch skymist

* Fixed minor bug in result.py, function make_plot_zoomed()

* Fixed minor bug in result.py, function make_plot_zoomed(), part 2

* Testing conection branch skymist Part 2

* New boolean value to identify rude events

* Exploring new contours for rude events Part 1

* Exploring new contours for rude events Part 2

* Exploring new contours for rude events Part 3

* Exploring new contours for rude events Part 4

* Exploring new contours for rude events Part 5

* Exploring new contours for rude events Part 5

* Exploring new contours for rude events Part 6

* Exploring new contours for rude events Part 7

* Implemented contours for rude events

* &lt;bot&gt; update requirements-examples.txt

* Solve issues for merge

* Moved circular_contour() to class level

* Missing self in circular_contour() on class level

* boolean parameter &#39;is_rude&#39; changed in &#39;circular&#39;. Added 50% and 90% radii as parameters.

* Improved handling of circular contours with numpy

* Fixed error in calling np.dstack()

* Fixing bugs after implementation of np.dstack()

* circular_contour() as static method

---------

Co-authored-by: Massimiliano Lincetto &lt;m.lincetto@gmail.com&gt;
Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`24ff08b`](https://github.com/icecube/skyreader/commit/24ff08b393df39da21b3d540c7e3197b7d400f29))


## v1.2.5 (2023-10-12)

###  

* &lt;bot&gt; update requirements-examples.txt ([`1458ac1`](https://github.com/icecube/skyreader/commit/1458ac1d2ce9ec33512a294bd86e34f823ddd375))

* Plotting functions cleanup (#21)

* deprecate log_func

* &lt;bot&gt; update requirements-examples.txt

* &lt;bot&gt; update requirements-tests.txt

* &lt;bot&gt; update requirements.txt

* deprecate upload-func

* deprecate further slack posting logic

* further remove deprecated logic

* assume we always save plots

* single source of truth for inches/dpi

* remove unused vars

* remove import io

* update example

* de-type plotting function

* some notes from mypy

* mypy readiness step

* deprecate test for npz format

* &lt;bot&gt; update setup.cfg

* &lt;bot&gt; update requirements-examples.txt

* &lt;bot&gt; update requirements-tests.txt

* &lt;bot&gt; update requirements.txt

* use new interface to colormaps

* list to array

* calculate area

* make it a static metod

* annotate mypy errors

* partial mypy compliance; annotate errors

---------

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`97dff8e`](https://github.com/icecube/skyreader/commit/97dff8efa79d5bd480f6e947f3fe47a87e565385))


## v1.2.4 (2023-08-11)

###  

* Check for numpy type when converting metadata (#19)

* only convert numpy types

* &lt;bot&gt; update requirements-examples.txt

* try isinstance solution

* cleanup comments

---------

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`d960116`](https://github.com/icecube/skyreader/commit/d960116901da45272529e69aa7b3787a6e5d16d9))


## v1.2.3 (2023-08-03)

###  

* Pin `python-semantic-release/python-semantic-release@v7.34.6` (#17) ([`c2f6cc4`](https://github.com/icecube/skyreader/commit/c2f6cc4b6bc5c489030b0d0123dde156cc79682d))

* &lt;bot&gt; update requirements.txt ([`8d5863f`](https://github.com/icecube/skyreader/commit/8d5863f4077404fc9e68b198199544fa3a36bb7c))

* &lt;bot&gt; update requirements-tests.txt ([`76dded5`](https://github.com/icecube/skyreader/commit/76dded5feeccf05babdfead4b1ca54251bb0e731))

* &lt;bot&gt; update requirements-examples.txt ([`cb4bd36`](https://github.com/icecube/skyreader/commit/cb4bd3625b092eaaa0aa8392b89fa666c6024657))

* Fix serialization (#16)

* Added minimal npz to JSON conversion script.

* CI testing for conversion.

* Comments and code cosmetics.



---------

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`5b756cc`](https://github.com/icecube/skyreader/commit/5b756cc271f958d6e1cf620f569a926af66c62d0))


## v1.2.2 (2023-07-13)

###  

* Empty Result Fix (No File) (#12) ([`0faec0d`](https://github.com/icecube/skyreader/commit/0faec0dd6c3c8fdcc43a03429ecddd98c5cd3de4))


## v1.2.1 (2023-07-13)

###  

* Empty Result Fix (#11)

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`a07c359`](https://github.com/icecube/skyreader/commit/a07c359190942de4a969e3f116d2cdbe3d93dac6))


## v1.2.0 (2023-06-07)

### [minor]

* Remove `do_disqualify_zero_energy_pixels` [minor] (#10) ([`3fe7d0d`](https://github.com/icecube/skyreader/commit/3fe7d0dcd8dc8ed53d3f2e99e8077bf6c4ca12db))


## v1.1.0 (2023-06-07)

### [minor]

* Add `rtol_per_field` for Result Comparison [minor] (#9)

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`b3cc988`](https://github.com/icecube/skyreader/commit/b3cc988c29e77119e03d4571a63e97eeeceb1c73))


## v1.0.2 (2023-05-04)

###  

* &lt;bot&gt; update requirements.txt ([`80f9203`](https://github.com/icecube/skyreader/commit/80f920397b36fd3e74134da331fce4c35a0d8474))

* &lt;bot&gt; update requirements-tests.txt ([`261f4b0`](https://github.com/icecube/skyreader/commit/261f4b0a62e730f5ca03f94e5e3ddcdf2984a1ff))

* &lt;bot&gt; update requirements-examples.txt ([`2eb6e46`](https://github.com/icecube/skyreader/commit/2eb6e46877279e769fceda9f47458bbeb2d2518e))

* Resolve imports in `result.py` (#6)

* fix imports

* &lt;bot&gt; update requirements-examples.txt

* &lt;bot&gt; update requirements-tests.txt

* &lt;bot&gt; update requirements.txt

* consolidate imports

---------

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`b1203e6`](https://github.com/icecube/skyreader/commit/b1203e66aca26a90e56e06919921078c2127a957))


## v1.0.1 (2023-04-20)

###  

* &lt;bot&gt; update setup.cfg ([`caf64f3`](https://github.com/icecube/skyreader/commit/caf64f3ed82a65d46fdc2875986c02250871150e))


## v1.0.0 (2023-04-20)

###  

* add scanner&#39;s files ([`cce323d`](https://github.com/icecube/skyreader/commit/cce323d2d9ac56476e4b75b0c1a61c5937adac4c))

* remove original files ([`638f09d`](https://github.com/icecube/skyreader/commit/638f09d0611ae7ba4bbb9ce844d5ffae07d08f6b))

* Add `--real-event` &amp; `--simulated-event` (required, mutex) (#122) ([`ca809c6`](https://github.com/icecube/skyreader/commit/ca809c677deca5e7413d97156fd1d1598720d23c))

* Additional SkyDriver Updates (#119)

Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`a47765e`](https://github.com/icecube/skyreader/commit/a47765e1db2b4a2ba6175762f76fc7460e4b43eb))

* Plotting functions in ScanResult (#82)

* working plotter in ScanResult

* refactor into a plotting_tools file

* add in the healpy plotting

* add meander dep

* &lt;bot&gt; update setup.cfg

* &lt;bot&gt; update requirements.txt

* tweak settings to plot 50,90, 3sigma wilks and set colormap max to 40

* don&#39;t set equal aspect for full sky plot

* &lt;bot&gt; update requirements.txt

* restore some more plot functionality

* &lt;bot&gt; update requirements.txt

* put plot catalog function back

* update plotting routines to use metadata if available

* tag nside in unique_id for filename

Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`4e38109`](https://github.com/icecube/skyreader/commit/4e38109a6b1f2fb0dea1b5056dd080506cd783e1))

* flip bounding box calculation of +/- dec error for Equatorial scan (#176) ([`3ea1c1b`](https://github.com/icecube/skyreader/commit/3ea1c1bcf733f97c7e7087f80ee2b5b46d2c6ec9))

* Shift ra to be near pi and fix printouts. see #96. ([`ef9177c`](https://github.com/icecube/skyreader/commit/ef9177c0464bd46d7aac51c96638e313b4571cdf))

* fix area calculation. ([`356fd5d`](https://github.com/icecube/skyreader/commit/356fd5d63353b407de29810c72b8054dfd692aad))

* Add `--real-event` &amp; `--simulated-event` (required, mutex) (#122) ([`7a7cd42`](https://github.com/icecube/skyreader/commit/7a7cd42e3fc12d001c38a35358f4a05940249f68))

* Additional SkyDriver Updates (#119)

Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`1937690`](https://github.com/icecube/skyreader/commit/19376903100a5e0755cab486e05d3d082a1dc6d0))

* Updates for SkyDriver (#109)

Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`b720272`](https://github.com/icecube/skyreader/commit/b7202727de710c84189bf15f1f4ae07607829c55))

* fix contour saving bug ([`0d5df63`](https://github.com/icecube/skyreader/commit/0d5df63fa19cf8a08ac8a7f6b095a941a4a479b1))

* Update a few settings to optimize contours and speed (#99)

* increase MinTimeWidth

* try to set quantileEpsilon and revert to float prec if it fails

* properly handle boost python exception

* tighten the tolerance and use fp32 precision for simplex

* remove the try-except for double prec

* coarser coarse steps

* &lt;bot&gt; update requirements.txt

* &lt;bot&gt; update setup.cfg

* add a function to skip most unhit doms but keep a sparse array to prevent local minima

* revert to previous base image

* reduce binsigma setting to weight timing more

* add fallback pos as an extra first guess

* switch colormap to plasma to see grid lines

Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`35550b6`](https://github.com/icecube/skyreader/commit/35550b6cc9b692ca7edf97740eacd6ae1e275afa))

* improve cartview bounds ([`bbf7d1c`](https://github.com/icecube/skyreader/commit/bbf7d1c3922f6525b1e70dd5df817933e14ff01f))

* separate icetray-dependent util functions (#100)

* separate icetray-dependent util functions from non-icetray dependent ones. Obviates need for icetray to make plots.

* actually add the simple.py

* save emuhi in json converter and add some querying functions

* &lt;bot&gt; update requirements.txt

* rename to icetrayless and add a few more query result funcs

* &lt;bot&gt; update setup.cfg

* improve names a bit

Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`5cf336c`](https://github.com/icecube/skyreader/commit/5cf336cd36e088c35de1e886c54f6412909cc3fe))

* revert formatter change in 35ce392 ([`63a8856`](https://github.com/icecube/skyreader/commit/63a8856bf9816bfd96e239fd428ad520ec96caf1))

* Fix ci (#98)

* patch for disk space issue

* localize plotting imports so we don&#39;t need icetray to check a scan result

* &lt;bot&gt; update setup.cfg

* &lt;bot&gt; update requirements.txt

Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`7f09de4`](https://github.com/icecube/skyreader/commit/7f09de48985692e940f49ed67829ecd7c7f00b19))

* Fix the contour bound calculation when ra is close to 0/2pi. (#96)

* Fix the contour bound calculation when ra is close to 0/2pi.

* bound latra

* &lt;bot&gt; update requirements.txt

Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`7c38e32`](https://github.com/icecube/skyreader/commit/7c38e321a1ad67aea7c6b1e33ce5644c45ccf030))

* Plotting functions in ScanResult (#82)

* working plotter in ScanResult

* refactor into a plotting_tools file

* add in the healpy plotting

* add meander dep

* &lt;bot&gt; update setup.cfg

* &lt;bot&gt; update requirements.txt

* tweak settings to plot 50,90, 3sigma wilks and set colormap max to 40

* don&#39;t set equal aspect for full sky plot

* &lt;bot&gt; update requirements.txt

* restore some more plot functionality

* &lt;bot&gt; update requirements.txt

* put plot catalog function back

* update plotting routines to use metadata if available

* tag nside in unique_id for filename

Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`1e18b69`](https://github.com/icecube/skyreader/commit/1e18b6979654055713a3458c6fd402a556609c69))

* A more generalized handling of metadata (#90)

* a more generalized handling of metadata

* add a function to check if ScanResult has minimum metadata and add to test ([`7c53fa1`](https://github.com/icecube/skyreader/commit/7c53fa152a230c29dfc806b73e50b324b879a441))

* Preserve useful metadata in the nside array (#88)

* preserve useful metadata in the nside array

* default optional arguments to None

* debug output for metadata

* workaround to preserve array metadata when saving ([`28fd254`](https://github.com/icecube/skyreader/commit/28fd254007874889f014c205c67b912cd212e3ef))

* Production Readiness (#68)

Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`75a5833`](https://github.com/icecube/skyreader/commit/75a5833f07c8ce5985e6710452b9771f6569fb70))

* Performance &amp; Configuration Upgrades (#59)

Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`1aee7a5`](https://github.com/icecube/skyreader/commit/1aee7a5b5978f32dc8cbcacb6baee3accff89dff))

### [major]

* Import Git History From Scanner &amp; Trim [major] ([`188b1fc`](https://github.com/icecube/skyreader/commit/188b1fc6f0f0e648c877157c3f681ba2426185f5))

### [minor]

* Perform scan in actual equatorial coordinates [minor] (#171)

* Perform scan in actual equatorial coordinates

* missed one

* update test files generated by flipping in dec

* catch an AttributeError when metadata doesn&#39;t exit

* &lt;bot&gt; update requirements-all.txt

* &lt;bot&gt; update requirements-client-starter.txt

* &lt;bot&gt; update requirements-gcp.txt

* &lt;bot&gt; update requirements-nats.txt

* &lt;bot&gt; update requirements-pulsar.txt

* &lt;bot&gt; update requirements-rabbitmq.txt

* &lt;bot&gt; update requirements.txt

* plotting fixes

* remove lat_offset reflection

---------

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`e84a3e4`](https://github.com/icecube/skyreader/commit/e84a3e4d54175dd8bad3bae5333ad3ec9aefd039))

* Reporter: Add Per-Nside Stats [minor] (#157)

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`b393800`](https://github.com/icecube/skyreader/commit/b3938002a632caa05a60493f1c41363142dfeb84))

* Perform scan in actual equatorial coordinates [minor] (#171)

* Perform scan in actual equatorial coordinates

* missed one

* update test files generated by flipping in dec

* catch an AttributeError when metadata doesn&#39;t exit

* &lt;bot&gt; update requirements-all.txt

* &lt;bot&gt; update requirements-client-starter.txt

* &lt;bot&gt; update requirements-gcp.txt

* &lt;bot&gt; update requirements-nats.txt

* &lt;bot&gt; update requirements-pulsar.txt

* &lt;bot&gt; update requirements-rabbitmq.txt

* &lt;bot&gt; update requirements.txt

* plotting fixes

* remove lat_offset reflection

---------

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`8831456`](https://github.com/icecube/skyreader/commit/883145656e95af30ab768a96c94d0d6ce837ed1f))

* Predictive Scanning &amp; Variable N-Sides [minor] (#158)

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`1d80ccc`](https://github.com/icecube/skyreader/commit/1d80cccb0e1344f5a3489c096747572ea1d70162))

* Reporter: Add Per-Nside Stats [minor] (#157)

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`2b9f924`](https://github.com/icecube/skyreader/commit/2b9f92401e921ed441d986a91eec86d0d4d4c272))


## v0.1.1 (2023-04-18)

###  

* &lt;bot&gt; update setup.cfg ([`1a5fc82`](https://github.com/icecube/skyreader/commit/1a5fc82a91f5774800a62aed20e4987805f714bc))


## v0.1.0 (2023-04-18)

### [minor]

* Migrate Files from Skymap Scanner [minor] (#4)

Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`96ae8d6`](https://github.com/icecube/skyreader/commit/96ae8d6cc7122d6bd655722e01203bfd0dc1c78d))


## v0.0.2 (2023-04-13)

###  

* &lt;bot&gt; update setup.cfg ([`2a6ab5c`](https://github.com/icecube/skyreader/commit/2a6ab5c2ac89b9ef3ad828d62ef932e75e0d3970))


## v0.0.1 (2023-04-13)

###  

* &lt;bot&gt; update requirements-tests.txt ([`98d2751`](https://github.com/icecube/skyreader/commit/98d2751442b8a82d9e2bb57712d30275620bfcd9))

* Package Infrastructure (#3)

Co-authored-by: github-actions &lt;github-actions@github.com&gt;
Co-authored-by: wipacdevbot &lt;developers@icecube.wisc.edu&gt; ([`d249bd9`](https://github.com/icecube/skyreader/commit/d249bd9abc20843c0d819abd6625ca3888f3d1e4))

* Initial commit ([`f928361`](https://github.com/icecube/skyreader/commit/f928361e5f8d0fec325f5848b5c1d41c04388ef5))
