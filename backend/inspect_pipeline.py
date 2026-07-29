import inspect

from diffusers import ZImagePipeline

print("=== Public methods/attributes on ZImagePipeline ===")
for name in dir(ZImagePipeline):
    if not name.startswith("_") or name == "__call__":
        print(name)

print("\n=== __call__ signature ===")
print(inspect.signature(ZImagePipeline.__call__))

if hasattr(ZImagePipeline, "encode_prompt"):
    print("\n=== encode_prompt signature ===")
    print(inspect.signature(ZImagePipeline.encode_prompt))
else:
    print("\nNo encode_prompt method found on ZImagePipeline.")
