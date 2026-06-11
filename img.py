import zlib
import os
import pefile

location = "/home/nikk/Downloads/0000029bfead495a003e43a7ab8406c6209ffb7d5e59dd212607aa358bfd66ea"
with open(location, "rb") as f:
    data = f.read()
decompressed_data = zlib.decompress(data)
pe = pefile.PE(data=decompressed_data)
print( {
    "entry_point": pe.OPTIONAL_HEADER.AddressOfEntryPoint,
    "image_base": pe.OPTIONAL_HEADER.ImageBase,
    "sections": [(section.Name.decode().rstrip('\x00'), section.VirtualAddress, section.SizeOfRawData) for section in pe.sections]
})

targetSection = None
LastSection = pe.sections[-1]
for section in pe.sections:
    sectionName = section.Name.decode().rstrip('\x00')
    if sectionName == ".text":
        targetSection = section
    #     startingAddess = section.PointerToRawData + section.SizeOfRawData - 24
    #     first5 = pe.get_data(startingAddess, 5)
    #     print(f"First 5 bytes of .text section: {first5}")
    #     context = first5
    #     newBytes = b'\x90\x90\x90\x90\x90'  # NOP instructions
    #     pe.set_bytes_at_offset(startingAddess, newBytes)
    #     print(section.get_data())
    # if sectionName == ".NewSec":
    #     section.Misc_VirtualSize += len(context)
    #     section.SizeOfRawData += len(context)
    #     pe.set_bytes_at_offset(section.PointerToRawData + section.SizeOfRawData - len(context), context)
    #     print(f"Added .NewSec section with size {len(context)} bytes.")
    #     print(context)
    #     print(section.get_data())

LastSection.Misc_VirtualSize += 5
LastSection.SizeOfRawData += 5

codeLoaderStart = targetSection.PointerToRawData + targetSection.SizeOfRawData - 24

targetContents = pe.get_data(codeLoaderStart, 5)
print(f"Original bytes at the end of .text section: {targetContents}")
pe.set_bytes_at_offset(LastSection.PointerToRawData + LastSection.SizeOfRawData - 5, targetContents)  # NOP instructions
print(LastSection.get_data())
for i in range(0, 5):
    originalByte = bytes([LastSection.get_data()[-5 + i]])
    pe.set_bytes_at_offset(codeLoaderStart + i, originalByte) 
    pe.set_bytes_at_offset(LastSection.PointerToRawData + LastSection.SizeOfRawData - 5 + i, b'\x00')  # NOP instruction 
print(f"Added 5 NOP instructions at the end of the last section. New size: {LastSection.get_data()} bytes.")
print(targetSection.get_data())
        
print( {
    "entry_point": pe.OPTIONAL_HEADER.AddressOfEntryPoint,
    "image_base": pe.OPTIONAL_HEADER.ImageBase,
    "sections": [(section.Name.decode().rstrip('\x00'), section.VirtualAddress, section.SizeOfRawData) for section in pe.sections]
})
        