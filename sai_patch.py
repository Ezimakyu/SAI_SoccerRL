# Some random patching to get code running apparently

import logging
import sai_mujoco.utils.v0.binding_utils as binding_utils

# Patching MjModel_v0 directly to handle slash prefixes globally

_original_body_name2id = binding_utils.MjModel_v0.body_name2id
_original_geom_name2id = binding_utils.MjModel_v0.geom_name2id
_original_joint_name2id = binding_utils.MjModel_v0.joint_name2id
_original_site_name2id = binding_utils.MjModel_v0.site_name2id

# Also patch accessor methods which might be raising KeyErrors directly
_original_body = getattr(binding_utils.MjModel_v0, 'body', None)
_original_geom = getattr(binding_utils.MjModel_v0, 'geom', None)
_original_joint = getattr(binding_utils.MjModel_v0, 'joint', None)
_original_site = getattr(binding_utils.MjModel_v0, 'site', None)

def _patched_body_name2id(self, name):
    try:
        return _original_body_name2id(self, name)
    except (ValueError, KeyError):
        # Try adding slash
        if not name.startswith("/"):
            try:
                return _original_body_name2id(self, "/" + name)
            except (ValueError, KeyError):
                pass
        # Try removing slash
        if name.startswith("/"):
            try:
                return _original_body_name2id(self, name[1:])
            except (ValueError, KeyError):
                pass
        raise

def _patched_geom_name2id(self, name):
    try:
        return _original_geom_name2id(self, name)
    except (ValueError, KeyError):
        # Try adding slash
        if not name.startswith("/"):
            try:
                return _original_geom_name2id(self, "/" + name)
            except (ValueError, KeyError):
                pass
        # Try removing slash
        if name.startswith("/"):
            try:
                return _original_geom_name2id(self, name[1:])
            except (ValueError, KeyError):
                pass
        raise

def _patched_joint_name2id(self, name):
    try:
        return _original_joint_name2id(self, name)
    except (ValueError, KeyError):
        # Try adding slash
        if not name.startswith("/"):
            try:
                return _original_joint_name2id(self, "/" + name)
            except (ValueError, KeyError):
                pass
        # Try removing slash
        if name.startswith("/"):
            try:
                return _original_joint_name2id(self, name[1:])
            except (ValueError, KeyError):
                pass
        raise

def _patched_site_name2id(self, name):
    try:
        return _original_site_name2id(self, name)
    except (ValueError, KeyError):
        # Try adding slash
        if not name.startswith("/"):
            try:
                return _original_site_name2id(self, "/" + name)
            except (ValueError, KeyError):
                pass
        # Try removing slash
        if name.startswith("/"):
            try:
                return _original_site_name2id(self, name[1:])
            except (ValueError, KeyError):
                pass
        raise

def _make_patched_accessor(original_descriptor):
    if original_descriptor is None:
        return None

    # Helper to perform the call with retries
    def _call_with_retries(func, name):
        try:
            return func(name)
        except (ValueError, KeyError):
            # Try adding slash
            if isinstance(name, str) and not name.startswith("/"):
                try:
                    return func("/" + name)
                except (ValueError, KeyError):
                    pass
            # Try removing slash
            if isinstance(name, str) and name.startswith("/"):
                try:
                    return func(name[1:])
                except (ValueError, KeyError):
                    pass
            raise

    # If it's a property, we return a new property
    if isinstance(original_descriptor, property):
        @property
        def _patched_prop(self):
            # Get the actual callable object from the original property
            original_callable = original_descriptor.__get__(self)
            
            # Return a wrapper that intercepts the call
            def _wrapper(name):
                return _call_with_retries(original_callable, name)
            return _wrapper
        return _patched_prop
    
    # If it's a standard method/function
    else:
        def _patched_method(self, name):
            # Bind the method to self if needed, or just call it passing self
            # original_descriptor is likely an unbound function here
            def _func(n):
                return original_descriptor(self, n)
            return _call_with_retries(_func, name)
        return _patched_method

_patched_body = _make_patched_accessor(_original_body)
_patched_geom = _make_patched_accessor(_original_geom)
_patched_joint = _make_patched_accessor(_original_joint)
_patched_site = _make_patched_accessor(_original_site)

print("[sai_patch] Applying GLOBAL sai_mujoco name patches...")
binding_utils.MjModel_v0.body_name2id = _patched_body_name2id
binding_utils.MjModel_v0.geom_name2id = _patched_geom_name2id
binding_utils.MjModel_v0.joint_name2id = _patched_joint_name2id
binding_utils.MjModel_v0.site_name2id = _patched_site_name2id

if _patched_body:
    binding_utils.MjModel_v0.body = _patched_body
if _patched_geom:
    binding_utils.MjModel_v0.geom = _patched_geom
if _patched_joint:
    binding_utils.MjModel_v0.joint = _patched_joint
if _patched_site:
    binding_utils.MjModel_v0.site = _patched_site

print("[sai_patch] Global patches applied.")
